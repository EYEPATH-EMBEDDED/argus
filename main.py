# main.py
import json
from fastapi import status
import base64
import asyncio
import logging
import cv2
import numpy as np
import httpx   # requests 대신 비동기 HTTP 라이브러리 사용
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ai_session.session import AISession

app = FastAPI()

# 활성 세션 저장소: user_id → AISession
sessions: dict[str, AISession] = {}

# 사용량 집계 정보 보낼 엔드포인트
USAGE_REPORT_URL = "http://10.0.20.167:8080/logs"
USER_INFO_URL    = "http://10.0.20.166:8080/users"


# ─── 로거 설정 ─────────────────────────────────────────────────────────────
logger = logging.getLogger("collision_ws")
# Uvicorn 기본 로깅 레벨을 가져오려면:
uvicorn_logger = logging.getLogger("uvicorn.error")
logger.handlers = uvicorn_logger.handlers
logger.setLevel(uvicorn_logger.level)
# ──────────────────────────────────────────────────────────────────────────


@app.websocket("/ws/collision")
async def collision_ws(websocket: WebSocket):
    print("▶ handshake 진입:", websocket.client)
    token = websocket.query_params.get("token")

    print("   token:", token)
    if not token:
        
        print("   ❌ 토큰 없음, 연결 거부")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # --- 2) 외부 /users API 호출로 토큰 검증 & 사용자 정보 획득 ---
    print("   🔑 사용자 조회 시도")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                USER_INFO_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0
            )
        print("   ↩️ /users 응답:", resp.status_code)
        if resp.status_code != 200:
            # 인증 실패
            print("   ❌ 인증 실패, 연결 거부")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        user = resp.json()  
        
        user_id   = user["userId"]
        print("   ✔ 인증 성공 user_id =", user_id)
    except Exception as e:
        # 네트워크 오류나 JSON 파싱 에러 등
        print("   💥 사용자 조회 중 예외:", e)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return
    
    print("   🟢 websocket.accept() 호출")
    await websocket.accept()

    # 새 세션 생성 or 기존 세션 가져오기
    session = sessions.setdefault(user_id, AISession(user_id))

    try:
        while True:
            text = await websocket.receive_text()
            msg = json.loads(text)

            if msg.get("type") == "image":
                # --- 이미지 디코딩 ---
                b64 = msg.get("data", "")
                img_data = base64.b64decode(b64)
                arr = np.frombuffer(img_data, dtype=np.uint8)

                # img == demo의 sample임 (line 30 참조)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(img, (1280, 720))
                if img is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "invalid image data"
                    })
                    continue

                # --- AI 예측 ---
                result = session.process_image(img)
                print(type(result), result)
                # --- 클라이언트로 전송 ---
                await websocket.send_json({
                    "type": "result",
                    "result" : int(result)
                })

            elif msg.get("type") == "end":
                # --- 세션 종료 처리 ---
                usage = session.get_usage()
                # 백그라운드로 비동기 POST
                asyncio.create_task(report_usage(usage))

                # 클라이언트에 최종 결과 전송
                await websocket.send_json({
                    "type": "ended",
                    "usage": usage
                })
                break

            else:
                # 잘못된 메시지 타입
                await websocket.send_json({
                    "type": "error",
                    "message": "unknown message type"
                })

    except WebSocketDisconnect:
        usage = session.get_usage()
        asyncio.create_task(report_usage(usage))
        

    finally:
        # 세션 정리
        if user_id in sessions:
            del sessions[user_id]
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"WebSocket close 예외 발생: {e}")   


async def report_usage(usage: dict):
    """
    사용량 집계를 USAGE_REPORT_URL에 비동기 POST 합니다.
    """
    async with httpx.AsyncClient() as client:
        try:
            await client.post(USAGE_REPORT_URL, json=usage, timeout=5.0)
            print("log push 완료")
        except Exception as e:
            # 로깅하거나, 재시도 로직을 추가하세요.
            print("Usage report failed:", e)