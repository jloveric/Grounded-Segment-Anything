from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from io import BytesIO
import PIL.Image as Image
import uvicorn
from grounded_sam_functions import GroundedSamExecutor

app = FastAPI()
gse = GroundedSamExecutor()


@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    print('file', file)
    try:
        contents = await file.read()
        #print('contents', contents)
        image = Image.open(BytesIO(contents)) 
        result = gse(input_image=image, text_prompt="houses")
        print('result', result.shape)

        #print('image', image)
        # Basic validation for PNG format
        if image.format != 'PNG':
            raise HTTPException(status_code=400, detail="Image must be in PNG format.")



        # You can add image processing steps here if needed  
        print('returning')
        return result.tolist()
        #return Response(content=result.tobytes(), media_type="application/octet-stream")
        #return Response(content=contents, media_type="image/png")

    except Exception as e:
        print('exception', e)
        raise HTTPException(status_code=500, detail="Error processing the image.")

# To start the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)