import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject, GLib
import numpy as np
import cv2
import sys
from models.demos.yolov4.demo import test_yolov4
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
import ttnn
import torch
import time
import math


FPS = 0


class GST:
    def __init__(self):
        self.device_id = 0
        self.device = ttnn.CreateDevice(
            self.device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2
        )
        ttnn.enable_program_cache(self.device)
        self.model = YOLOv4PerformantRunner(self.device)
        # self.model.initialize_yolov4_trace_2cqs_inference(self.device)

    def release_trace(self):
        self.model.release()

    def plot_boxes_cv2(self, img, boxes, savename=None, class_names=None, color=None):
        img = np.copy(img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            bbox_thick = int(0.6 * (height + width) / 600)
            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print("%s: %f" % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                msg = str(class_names[cls_id]) + " " + str(round(cls_conf, 3))
                t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
                c1, c2 = (x1, y1), (x2, y2)
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                # print("#####################################",img)
                # print("#####################################",c1,c2,t_size,c3)
                cv2.rectangle(img, (x1, y1), (int(np.float32(c3[0])), int(np.float32(c3[1]))), rgb, -1)
                img = cv2.putText(
                    img,
                    msg,
                    (c1[0], int(np.float32(c1[1] - 2))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA,
                )

            img = cv2.rectangle(img, (x1, y1), (int(x2), int(y2)), rgb, bbox_thick)
        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, img)
        return img

    def load_class_names(self, namesfile):
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

    def post_processing(self, img, conf_thresh, nms_thresh, output):
        # [batch, num, 1, 4]
        box_array = output[0]
        # [batch, num, num_classes]
        confs = output[1].float()

        t1 = time.time()

        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        t2 = time.time()

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = self.nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )

            bboxes_batch.append(bboxes)

        t3 = time.time()

        print("-----------------------------------")
        print("       max and argmax : %f" % (t2 - t1))
        print("                  nms : %f" % (t3 - t2))
        print("Post processing total : %f" % (t3 - t1))
        print("-----------------------------------")

        return bboxes_batch

    def process_frame(self, sample):
        """Processes a GStreamer sample and converts it to a NumPy array."""
        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        format_str = structure.get_value("format")

        ret, map_info = buf.map(Gst.MapFlags.READ)
        if not ret:
            print("Failed to map buffer")
            return None

        if format_str == "RGB":
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)
        elif format_str == "BGR":
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        elif format_str == "GRAY8":
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width)
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
        elif format_str == "YUY2":
            yuy2_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 2)
            frame_data = cv2.cvtColor(yuy2_data, cv2.COLOR_YUV2RGB_YUYV)
        elif format_str == "NV12":
            y_size = width * height
            y_data = np.frombuffer(map_info.data, dtype=np.uint8, count=y_size).reshape(height, width)
            uv_data = np.frombuffer(map_info.data, dtype=np.uint8, offset=y_size).reshape(height // 2, width // 2, 2)
            frame_data = cv2.cvtColorTwoPlane(y_data, uv_data, cv2.COLOR_YUV2RGB_NV12)
        elif format_str == "I420_10LE":
            data = np.frombuffer(map_info.data, dtype=np.uint16)
            y_size = width * height
            y_data = data[:y_size].reshape(height, width)
            u_data = data[y_size : y_size + (width // 2) * (height // 2)].reshape(height // 2, width // 2)
            v_data = data[y_size + (width // 2) * (height // 2) :].reshape(height // 2, width // 2)

            y_data = (y_data >> 2).astype(np.uint8)
            u_data = (u_data >> 2).astype(np.uint8)
            v_data = (v_data >> 2).astype(np.uint8)

            i420_data = np.zeros((height + height // 2, width), dtype=np.uint8)
            i420_data[:height, :width] = y_data
            i420_data[height : height + height // 2, ::2] = u_data
            i420_data[height : height + height // 2, 1::2] = v_data

            frame_data = cv2.cvtColor(i420_data, cv2.COLOR_YUV2RGB_I420)

        if len(frame_data.shape) == 2:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)

        if len(frame_data.shape) == 3 and frame_data.shape[2] == 4:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)

        if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
            if format_str != "BGR":
                frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        buf.unmap(map_info)
        return frame_data, buf.pts, buf.duration  # return frame data, and the buffer presentation timestamp (pts)

    def on_new_sample(self, sink):
        """Callback function for the appsink element."""
        sample = sink.emit("pull-sample")
        frame, pts, duration = self.process_frame(sample)
        global FPS

        if frame is not None:
            # frame_number = pts / 1000000000.0 # Convert nanoseconds to seconds.
            frame_number = pts / 3000000000.0  # Convert nanoseconds to micro seconds.
            if frame_number.is_integer():
                # if frame_number:
                img = frame
                # print(f"Frame number (approximate): {frame_number}, Frame shape: {frame.shape}")
                if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
                    img = torch.from_numpy(img.transpose(0, 1, 2)).float().div(255.0).unsqueeze(0)
                elif type(img) == np.ndarray and len(img.shape) == 4:
                    img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
                else:
                    exit()

                t1 = time.time()
                result_boxes, result_confs = self.model.run(img)
                t2 = time.time()
                print("##########################FPS#############################", (t2 - t1))
                FPS = FPS + (t2 - t1)

                ## Giraffe image detection
                conf_thresh = 0.3
                nms_thresh = 0.4
                output = [result_boxes.to(torch.float16), result_confs.to(torch.float16)]

                boxes = self.post_processing(img, conf_thresh, nms_thresh, output)
                namesfile = "models/demos/yolov4/resources/coco.names"
                class_names = self.load_class_names(namesfile)
                # img = cv2.imread(imgfile)
                img = frame
                frame = self.plot_boxes_cv2(img, boxes[0], f"ssinghal_{frame_number}.jpg", class_names=class_names)
                # test_yolov4(device,frame,frame_number)
            # Example: cv2.imshow("Frame", frame)
            # cv2.waitKey(1)
            data = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
            buf.pts = pts
            buf.duration = duration
            # We push to appsrc directly from the processing callback,
            # so no need for a separate list.
            # retval = appsrc.emit("push-buffer", buf)
            # if retval != Gst.FlowReturn.OK:
            #    print("Error pushing buffer:", retval)
            return Gst.FlowReturn.OK

    def run_pipeline(self, file_path):
        """Creates and runs the GStreamer pipeline."""
        Gst.init(None)

        pipeline_str = f"""
            filesrc location="{file_path}" !
            decodebin !
            videoconvert !
            videoscale !
            video/x-raw,width=320,height=320,format=BGR !
            appsink name=appsink
        """
        # appsrc name=appsrc !

        # model = Yolov4Trace2CQ()
        # model.initialize_yolov4_trace_2cqs_inference(device)
        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("appsink")
        appsink.set_property("emit-signals", True)
        # appsink.connect("new-sample", on_new_sample, appsrc)

        # appsrc = pipeline.get_by_name("appsrc")
        # appsrc.set_property("emit-signals", True)
        appsink.connect("new-sample", self.on_new_sample)
        # appsrc.connect("new-sample", on_new_sample, appsrc)

        # appsink.link(appsrc)

        # self.release_trace()
        pipeline.set_state(Gst.State.PLAYING)

        bus = pipeline.get_bus()
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

        pipeline.set_state(Gst.State.NULL)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"Error: {err}, {debug}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    gst = GST()
    ts = time.time()
    gst.run_pipeline(file_path)
    gst.release_trace()
    te = time.time()
    print("Final avg FPS", FPS / 300)
    print("Final avg time", te - ts)
