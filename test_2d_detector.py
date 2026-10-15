import asyncio
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

async def main():
    # Path to the model's configuration file
    config_file = 'configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'

    # Path to the model's checkpoint file (the weights)
    checkpoint_file = 'weights/maskrcnn/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    # Path to the image you want to test
    img_path = '000100.png'

    # Set the device to use (e.g., 'cuda:0' for GPU or 'cpu')
    device = 'cuda:0'

    print("Initializing the model...")
    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=device)
    print("Model initialized successfully!")

    print(f"Performing inference on {img_path}...")
    # Test a single image
    result = inference_detector(model, img_path)
    print("Inference complete!")

    # Show the results
    print("Displaying results... Close the image window to exit.")
    show_result_pyplot(model, img_path, result, score_thr=0.5)

if __name__ == '__main__':
    asyncio.run(main())