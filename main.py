iimport cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

def smooth_fill_balloons(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 60, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        dilated_mask = cv2.dilate(mask, kernel, iterations=15)
        border = cv2.bitwise_and(dilated_mask, dilated_mask, mask=cv2.bitwise_not(mask))

        avg_color = cv2.mean(image, mask=border)[0:3]

        cv2.drawContours(result, [cnt], -1, avg_color, -1)

    return result


@app.post("/clean")
async def clean_balloons(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    cleaned = smooth_fill_balloons(img_np)

    _, buffer = cv2.imencode(".png", cleaned)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


# ⬇⬇⬇ SERVIDOR UVICORN NECESARIO PARA RENDER ⬇⬇⬇
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
