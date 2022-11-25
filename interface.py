import cv2
import time
import av
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
tf.gfile = tf.io.gfile
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Set page configs.
st.set_page_config(page_title="Sign Language Detection", layout="centered")

#-----------------------------------------------------------------------------

# # Path of the pre-trained TF model
MODEL_DIR = r"./trained_model/saved_model"

# # Path of the LabelMap file
PATH_TO_LABELS = r"./trained_model/label_map.pbtxt"

# Decision Threshold
MIN_THRESH = float(0.60)

@st.cache(allow_output_mutation=True)
def load_model():
     print('Loading model...', end='')
     start_time = time.time()
     # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
     detect_fn = tf.saved_model.load(MODEL_DIR)
     end_time = time.time()
     elapsed_time = end_time - start_time
     print('Done! Took {} seconds'.format(elapsed_time))
     return detect_fn

# load model
detect_fn = load_model()

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 50px;font-weight: 350;font-family:Cursive "> Sign Language Detection </p>'
st.markdown(title, unsafe_allow_html=True)

# -------------Sidebar Section------------------------------------------------

with st.sidebar:

    title = '<p style="font-size: 25px;font-weight: 550;">Sign Language Detection</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Detection Mode", ('Image Upload',
                                                   'Webcam Image Capture',
                                                   'Webcam Realtime'), index=0)
    if mode == 'Image Upload':
        detection_mode = mode
    elif mode == 'Video Upload':
        detection_mode = mode
    elif mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Realtime':
        detection_mode = mode

# -------------Image Upload Section------------------------------------------------

if detection_mode == "Image Upload":
    
    st.markdown("&nbsp; Upload your Image below and our ML model will "
                "detect signs inside the Image", unsafe_allow_html=True)

    # Example Image
    st.image(image="./imgs/collage.jpg")
    st.markdown("</br>", unsafe_allow_html=True)

    # Upload the Image
    content_image = st.file_uploader(
        "Upload Content Image (PNG & JPG images only)", type=['png', 'jpg', 'jpeg'])

    st.markdown("</br>", unsafe_allow_html=True)
    st.warning('NOTE : You need atleast Intel i3 with 8GB memory for proper functioning of this application. ' +
            ' All Images are resized to 640x640')

    if content_image is not None:

        with st.spinner("Scanning the Image...will take few secs"):

            content_image = Image.open(content_image)

            content_image = np.array(content_image)

            # Resize image to 640x640
            content_image = cv2.resize(content_image, (640,640))

            # ---------------Detection Phase-------------------------

            # LOAD LABEL MAP DATA FOR PLOTTINg
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(content_image)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # Detect Objects
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_with_detections = content_image.copy()

            # Detected classes
            detected_classes = detections['detection_classes']
            scores = detections['detection_scores']

            print("Detected Classes : ", detected_classes)
            print("Scores : ", scores)

            # ---------------Drawing Phase-------------------------

            classes = {1: "food",
                    2: "yes",
                    3: "no",
                    4: "hello",
                    5: "thank_you"}
        
            responses= {
                "hello":"Hi, Nice to meet you !",
                "yes":"Great !",
                "no":"It's okay, Alright !",
                "thank_you":"Welcome !",
                "food":"Wait...",
            }
            
            response_imgs = {
                "hello":"./imgs/nice to meet you.png",
                "yes":"./imgs/great.png",
                "no":"./imgs/its okay.png",
                "thank_you":"./imgs/welcome.png",
                "food":"./imgs/wait.png",
            }
        

            # Find indexes with scores greater than the MIN_THRESH
            score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
            detected_classes = [detected_classes[idx] for idx in score_indexes]
            # Replace numbers with class names
            detected_classes = [classes.get(i) for i in detected_classes]

            if len(detected_classes)!=0:
            
                # Draw the bounding boxes with probability score
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=MIN_THRESH,
                    agnostic_mode=False)

                print('Done')

                if image_with_detections is not None:
                    # some baloons
                    st.balloons()

                col1, col2 = st.columns(2)
                with col1:
                    # Display the output
                    st.image(image_with_detections)
                with col2:
                    st.markdown("</br>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Detected : {detected_classes[0]} </h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Response : {responses[detected_classes[0]]} </h5>", unsafe_allow_html=True)
                    
                    # Display the response image
                    st.image(image=response_imgs[detected_classes[0]])
                    
                    st.markdown(
                        "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                    # convert to pillow image
                    img = Image.fromarray(image_with_detections)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")
        
            else:
                st.markdown(f"<h5> No Signs Found inside Image..Try another Image ! </h5>", unsafe_allow_html=True)      
        
# -------------Webcam Image Capture Section------------------------------------------------

if detection_mode == "Webcam Image Capture":

    st.info("NOTE : In order to use this mode, you need to give webcam access.")

    img_file_buffer = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                      help="Make sure you have given webcam permission to the site")

    if img_file_buffer is not None:

        with st.spinner("Detecting Signs ..."):
            
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)

            # To convert PIL Image to numpy array:
            img = np.array(img)
            
            # Resize image to 640x640
            content_image = cv2.resize(img, (640,640))

            # ---------------Detection Phase-------------------------

            # LOAD LABEL MAP DATA FOR PLOTTINg
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(content_image)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # Detect Objects
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_with_detections = content_image.copy()

            # Detected classes
            detected_classes = detections['detection_classes']
            scores = detections['detection_scores']

            print("Detected Classes : ", detected_classes)
            print("Scores : ", scores)

            # ---------------Drawing Phase-------------------------

            classes = {1: "food",
                    2: "yes",
                    3: "no",
                    4: "hello",
                    5: "thank_you"}
        
            responses= {
                "hello":"Hi, Nice to meet you !",
                "yes":"Great !",
                "no":"It's okay, Alright !",
                "thank_you":"Welcome !",
                "food":"Wait...",
            }
            
            response_imgs = {
                "hello":"./imgs/nice to meet you.png",
                "yes":"./imgs/great.png",
                "no":"./imgs/its okay.png",
                "thank_you":"./imgs/welcome.png",
                "food":"./imgs/wait.png",
            }
        

            # Find indexes with scores greater than the MIN_THRESH
            score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
            detected_classes = [detected_classes[idx] for idx in score_indexes]
            # Replace numbers with class names
            detected_classes = [classes.get(i) for i in detected_classes]

            if len(detected_classes)!=0:
            
                # Draw the bounding boxes with probability score
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=MIN_THRESH,
                    agnostic_mode=False)

                print('Done')

                if image_with_detections is not None:
                    # some baloons
                    st.balloons()

                col1, col2 = st.columns(2)
                with col1:
                    # Display the output
                    st.image(image_with_detections)
                with col2:
                    st.markdown("</br>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Detected : {detected_classes[0]} </h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5> Response : {responses[detected_classes[0]]} </h5>", unsafe_allow_html=True)
                    
                    # Display the response image
                    st.image(image=response_imgs[detected_classes[0]])
                    
                    st.markdown(
                        "<b> Your Image is Ready ! Click below to download it. </b> ", unsafe_allow_html=True)

                    # convert to pillow image
                    img = Image.fromarray(image_with_detections)
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    st.download_button(
                        label="Download image",
                        data=buffered.getvalue(),
                        file_name="output.png",
                        mime="image/png")
        
            else:
                st.markdown(f"<h5> No Signs Found inside Image..Try another Image ! </h5>", unsafe_allow_html=True)    

# -------------Webcam Realtime Section------------------------------------------------

if detection_mode == "Webcam Realtime":

    st.warning("NOTE : In order to use this mode, you need to give webcam access. "
               "After clicking 'Start' , it takes about 10-20 seconds to ready the webcam.")

    spinner_message = "Wait a sec, getting some things done..."

    with st.spinner(spinner_message):

        class VideoProcessor:

            def recv(self, frame):
                # convert to numpy array
                
                frame = frame.to_ndarray(format="bgr24")

                # Resize image to 640x640
                content_image = cv2.resize(frame, (640,640))

                # ---------------Detection Phase-------------------------

                # LOAD LABEL MAP DATA FOR PLOTTINg
                category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)

                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(content_image)

                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                # Detect Objects
                detections = detect_fn(input_tensor)

                # All outputs are batches tensors.
                # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                # We're only interested in the first num_detections.
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                image_with_detections = content_image.copy()

                # Detected classes
                detected_classes = detections['detection_classes']
                scores = detections['detection_scores']

                print("Detected Classes : ", detected_classes)
                print("Scores : ", scores)
                
                # ---------------Drawing Phase-------------------------

                classes = {1: "food",
                        2: "yes",
                        3: "no",
                        4: "hello",
                        5: "thank_you"}
                
                # Find indexes with scores greater than the MIN_THRESH
                score_indexes = [idx for idx, element in enumerate(scores) if element > MIN_THRESH]
                detected_classes = [detected_classes[idx] for idx in score_indexes]
                # Replace numbers with class names
                detected_classes = [classes.get(i) for i in detected_classes]

                if len(detected_classes)!=0:
            
                    # Draw the bounding boxes with probability score
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=MIN_THRESH,
                        agnostic_mode=False)

                    print('Done')

                frame = av.VideoFrame.from_ndarray(image_with_detections, format="bgr24")

                return frame

        webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                        rtc_configuration=RTCConfiguration(
                            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

