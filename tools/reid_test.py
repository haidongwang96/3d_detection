import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import track, tracker



extractor = torchreid.utils.FeatureExtractor(
    model_name='mobilenetv2_x1_0',
    model_path="../data/weights/mobilenetv2_1dot0_market.pth.tar",
    device='cuda'
)

from deep_sort_realtime.deepsort_tracker import DeepSort
"""
tracker = DeepSort(max_age=5)
bbs = object_detector.detect(frame) # your own object detection
object_chips = chipper(frame, bbs) # your own logic to crop frame based on bbox values
embeds = embedder(object_chips) # your own embedder to take in the cropped object chips, and output feature vectors
tracks = tracker.update_tracks(bbs, embeds=embeds) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class ), also, no need to give frame as your chips has already been embedded
for track in tracks:
    if not track.is_confirmed():
        continue
    track_id = track.track_id
    ltrb = track.to_ltrb()
"""