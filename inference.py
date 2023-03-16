from lib import *
from model import SSD
from transform import DataTransform


classes = ["500","1k","2k","5k","10k","20k","50k","100k","200k","500k"]

cfg = {
    "num_classes": 11, #VOC data include 20 class + 1 background class
    "input_size": 300, #SSD300
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Tỷ lệ khung hình cho source1->source6`
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Size of default box
    "max_size": [60, 111, 162, 213, 264, 315], # Size of default box
    "aspect_ratios": [[2], [2,3], [2,3], [2,3], [2], [2]]
}

net = SSD(phase="inference", cfg=cfg)
net_weights = torch.load("./data/weights/ssd300_100.pth", map_location={"cuda:0":"cpu"})
net.load_state_dict(net_weights)

def show_predict(img_file_path):
    img = cv2.imread(img_file_path)
    # cap = cv2.VideoCapture(video_path)

    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    phase = "val"
    img_tranformed, boxes, labels = transform(img, phase, "", "")
    img_tensor = torch.from_numpy(img_tranformed[:,:,(2,1,0)]).permute(2,0,1)

    net.eval()
    input = img_tensor.unsqueeze(0) #(1, 3, 300, 300)
    output = net(input)

    plt.figure(figsize=(10, 10))
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.3:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          colors[i%3], 2
                          )
            display_text = "%s: %.2f"%(classes[i-1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# def show_predict(video_file_path, output_file_path):
#     cap = cv2.VideoCapture(video_file_path)

#     color_mean = (104, 117, 123)
#     input_size = 300
#     transform = DataTransform(input_size, color_mean)

#     phase = "val"
#     net.eval()

#     # Set up video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     writer = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         img_tranformed, boxes, labels = transform(frame, phase, "", "")
#         img_tensor = torch.from_numpy(img_tranformed[:,:,(2,1,0)]).permute(2,0,1)

#         input = img_tensor.unsqueeze(0) #(1, 3, 300, 300)
#         output = net(input)

#         plt.figure(figsize=(10, 10))
#         colors = [(255,0,0), (0,255,0), (0,0,255)]
#         font = cv2.FONT_HERSHEY_SIMPLEX

#         detections = output.data #(1, 21, 200, 5) 5: score, cx, cy, w, h
#         scale = torch.Tensor(frame.shape[1::-1]).repeat(2)

#         for i in range(detections.size(1)):
#             j = 0
#             while detections[0, i, j, 0] >= 0.3:
#                 score = detections[0, i, j, 0]
#                 pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
#                 cv2.rectangle(frame,
#                               (int(pt[0]), int(pt[1])),
#                               (int(pt[2]), int(pt[3])),
#                               colors[i%3], 2
#                               )
#                 display_text = "%s: %.2f"%(classes[i-1], score)
#                 cv2.putText(frame, display_text, (int(pt[0]), int(pt[1])),
#                     font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#                 j += 1

#         writer.write(frame) # Write the frame to output file
        
#         cv2.imshow("Result", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     writer.release() # Release the video writer
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    # video_path = "./data/100_2.mp4"
    # output_path = "./data/100_2_output.mp4"
    image_path = "./data/500k_fake.jpg"
    show_predict(image_path)