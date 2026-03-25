import os
from ultralytics import YOLO

def main():
    print(f"Current working directory: {os.getcwd()}")
    
    model = YOLO('yolov8n-obb.pt') 

    results = model.train(
        data='config/cargo_dataset.yaml', 
        epochs=100,                       
        imgsz=640,                        
        batch=16,                         
        device='0',                       
        project='weights',                
        name='cargo_obb_run',             
        workers=4,                        
        patience=20,                      
    )
    
    print("Training finished. Check the 'weights/cargo_obb_run' directory for results and weights.")

if __name__ == '__main__':
    main()
