************************************************************************************************************
Copyright 2023 PROJECT GUIDE AND TEAM.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*************************************************************************************************************
 EMAIL  : JKANANT@GMAIL.COM
*************************************************************************************************************
File/Directories 
*************************************************************************************************************
-->person_counter.py
-->run_ip.py(run via CCTV)
-->person_detection_videos.py(video detection LOCAL)
-->person_detection_images.py(images detection LOCAL)

	/dataset 
		--> Videos 
		-->images
		-->frames(optional)

	/models/object detectors
		-->SSD's (TESLA) --> MobiNetSSD + resnetSSD
		-->OPENCV -->SSD's
		-->Faster R-CNN Model
		-->YOLO COCO model
	/records
		-->accuracy/output/end points
		-->tensorboards
		-->api's for front end
	/utils
		-->centroidtracker.py
		-->centroidtracker1.py
		-->config.py
		-->mailer.py(optional)
		-->people_counter.py
		-->thread.py
		-->tracable.py
		-->trackableobject.py
	/openvino_env(optional! required to run specific script)
	

