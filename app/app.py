from fastapi import FastAPI, File, UploadFile, Response,Query,Request,status,WebSocket,Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse , JSONResponse
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
from typing import List


import uvicorn
import cv2  
import numpy as np
import asyncio
import json
import re
import tempfile
import shutil
import os
import warnings
import tensorflow as tf
import io


from app.src.sign_to_text import video_prediction
from app.src.sign_to_text_frame import image_prediction



app=FastAPI(title="American Sign Language detection ",
    description="FastAPI",
    version="0.104.1")

# Allow all origins (replace * with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 



@app.get("/")
async def root():
  return {"Fast API":"API is woorking"}


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out info, 2 = filter out warnings, 3 = filter out errors
warnings.filterwarnings("ignore")

################## WebSocket connection manager #######################
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

############ sign To text #########################
@app.websocket("/ws/image_frame")
async def websocket_endpoint_image(websocket: WebSocket):
    await manager.connect(websocket)
    #final sent create variable

    temp_charecter=[]
    final_charecter=[]
    words=[]
    charecter=''
    sent=''
    
    try:
        while True:
            data = await websocket.receive_bytes()

            #nparr = np.frombuffer(data, np.uint8)
            #frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
            # Display the frame in a window (for server-side viewing)
            #cv2.imshow("Client Video Stream", frame)
            # Check for reset command
            
               # Check if the received data is binary
            try:
                command = data.decode('utf-8')
            
                # Handle command here (e.g., start/stop processing)
                if command.startswith("RESET"):
                    command =  command[ : ].strip()
                    print(f"Received command from : {command}")
                    temp_charecter.clear()
                    final_charecter.clear()
                    words.clear()
                    sent = ''
                    image_prediction(data=data,kill=1)
                    result = json.dumps({"message": "Predictions have been reset."})
                    await websocket.send_text(result)
                    await asyncio.sleep(0.1)
                    continue
                    
                elif command.startswith("STOP"):
                    command =  command[ : ].strip()
                    print(f"Received command from : {command}")
                    temp_charecter.clear()
                    final_charecter.clear()
                    words.clear()
                    sent = ''
                    image_prediction(data=data,kill=1)
                    result = json.dumps({"message": "Programe have been stop"})
                    await websocket.send_text(result)
                    await asyncio.sleep(0.1)
                    break
            
            except Exception as e:
                #print(f"Exception error: {e}")
                pass

            #Get the model prediction
            charecter = image_prediction(data=data, kill=0)

            if charecter :
                # filter the final Character
                if len(temp_charecter) !=0 :
                    if temp_charecter[-1] == charecter:
                        temp_charecter.append(charecter)
                        if len(temp_charecter) >= (5):
                            final_charecter.append(temp_charecter[-1])
                            temp_charecter.clear()
                    else:
                        temp_charecter.clear()
                else:
                    temp_charecter.append(charecter)

            #create the words
            word=''
            if len(final_charecter) !=0:
                for i in range(len(final_charecter)):
                    if final_charecter[i]=='SPACE':
                        words.append(word)
                        final_charecter.clear()
                        break
                    else:
                        word=word+final_charecter[i]
            else:
                pass

            #create the final sentence
            if len(words) != 0 :
                sent = " ".join(words)
                #print("sentence ",sent)
                final_sent=f"final sentence :{sent}"
            else:
                pass

            result= json.dumps({"present charecter":charecter,
                                "final charecter":final_charecter,
                                "word":word,
                                "words":words,
                                "sentence":sent})

            # Send each prediction from the generator
            await websocket.send_text(result)
            await asyncio.sleep(0.1)
        
    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        print(f"Error: {e}")  


@app.websocket("/ws/endpoint_check")
async def websocket_endpoint_check(websocket: WebSocket):
    await websocket.accept()

    t=0
    try:
        while True:
            data = await websocket.receive_bytes()  # Receive video frame as bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV format (image)
            #print("frame ok")
            
            if frame is not None:
                t+=1
                # Perform prediction on the frame
                prediction = str(t)               
                # Display the frame in a window (for server-side viewing)
                cv2.imshow("Client Video Stream", frame)
                # Send each prediction from the generator
                await websocket.send_text(prediction)
                
            else:
                print("frame null")

            # Exit if 'q' is pressed (so the server can terminate the display loop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except WebSocketDisconnect:
        print("Client disconnected")
    
    finally:
        cv2.destroyAllWindows()  # Close the OpenCV window when the connection is closed


@app.websocket("/ws/predict_video")
async def predict(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    try:
        while True:
            # Simulate getting input data for the model (replace with actual input)
            
            for prediction in video_prediction(video_path=0,wait_second=0.5):
            # Send each prediction from the generator
              await websocket.send_text(prediction)
              await asyncio.sleep(0.1) 
            
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()


@app.get("/stream_video")
def stream_video(link:int):
    headers = {
        "content-type": "text/event-stream",
    }

    return StreamingResponse(
        content=video_prediction(video_path=link,wait_second=0.5),
        headers=headers,
        status_code=200,
        media_type="text/event-stream",
    )





  



