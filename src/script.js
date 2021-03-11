import * as munkres from "munkres-js";

const status = document.getElementById("status");
status.innerText = "Loaded TensorFlow.js - version: " + tf.version.tfjs;

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");

let model = undefined;
let id = 0;

let children = [];
let detections = [];
let trackers = new Map();
let prevBbox;
let prevContext;

// Before we can use COCO-SSD class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment
// to get everything needed to run.
// Note: cocoSsd is an external object loaded from our index.html
// script tag import so ignore any warning in Glitch.
cocoSsd.load().then(function (loadedModel) {
	model = loadedModel;
	demosSection.classList.remove("invisible");
});

function getUserMediaSupported() {
	return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (getUserMediaSupported()) {
	enableWebcamButton.addEventListener("click", enableCam);
} else {
	console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
	if (!model) {
		return;
	}

	event.target.classList.add("removed");

	const constraints = {
		video: true,
	};

	navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
		video.srcObject = stream;
		var context = canvas.getContext("2d");
		context.drawImage(video, 0, 0, 100, 100 / (4 / 3));
		prevContext = canvas.getContext("2d");

		video.addEventListener("loadeddata", predictWebcam);
	});
}

let count = 0;
function predictWebcam() {

	detections = [];

	// Now let's start classifying a frame in the stream.
	model.detect(video).then(function (predictions) {
		// Remove any highlighting we did previous frame.
		for (let i = 0; i < children.length; i++) {
			liveView.removeChild(children[i]);
		}
		children.splice(0);

		// Now lets loop through predictions and draw them to the live view if
		// they have a high confidence score.
		for (let n = 0; n < predictions.length; n++) {
			// If we are over 66% sure we are sure we classified it right, draw it!
			if (predictions[n].score > 0.66) {
				const p = document.createElement("p");
				p.innerText =
					predictions[n].class +
					" - with " +
					Math.round(parseFloat(predictions[n].score) * 100) +
					"% confidence.";
				p.style =
					"margin-left: " +
					predictions[n].bbox[0] +
					"px; margin-top: " +
					(predictions[n].bbox[1] - 10) +
					"px; width: " +
					(predictions[n].bbox[2] - 10) +
					"px; top: 0; left: 0;";

				const highlighter = document.createElement("div");
				highlighter.setAttribute("class", "highlighter");
				highlighter.style =
					"left: " +
					predictions[n].bbox[0] +
					"px; top: " +
					predictions[n].bbox[1] +
					"px; width: " +
					predictions[n].bbox[2] +
					"px; height: " +
					predictions[n].bbox[3] +
					"px;";

				liveView.appendChild(highlighter);
				liveView.appendChild(p);
				children.push(highlighter);
				children.push(p);

				if (predictions[n].class === "person") {
					detections.push(predictions[n].bbox);
				}
			}
		}

		let ioUMatrix = [[]];
		let i = 0;
		Array.from(trackers.keys()).forEach((id) => {
			// dear alex says that for const of is superior but :p
			// need to get the pixels from the video frame at the bbox

			var context = canvas.getContext("2d");
			context.drawImage(video, 0, 0, 100, 100 / (4 / 3));
			// var frame = context.getImageData(x, y, width, height).data;

			let prevBbox = trackers.get(id);
			// let prevBboxFrame = prevContext.getImageData(...prevBbox);
			let j = 0;
			detections.forEach((bbox) => {
				// let bboxFrame = context.getImageData(...bbox);
				ioUMatrix[(i, j)] = IoU(bbox, prevBbox);
				j++;
			});
			
			i++;
			
			let associations = munkres([ioUMatrix]);

			// match bbox with the detections
			// update trackers
			associations.forEach(match => {
				trackers.set(match[0], detections[match[1]]);
				detections[match[1]] = [];
			});
		});
		
		// add new trackers
		detections.forEach((bbox) => {
			if (bbox.length === 4 ) {
				trackers.set(id, bbox);
				id += 1;
			}
		});
		

		console.log('trackers', trackers);
			
		// Call this function again to keep predicting when the browser is ready.
		window.requestAnimationFrame(predictWebcam);

	});
}

function IoU(bboxA, bboxB) {
	// https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections/41660682
	let xA = Math.max(bboxA[0], bboxB[0])
	let yA = Math.max(bboxA[1], bboxB[1])
	let xB = Math.min(bboxA[2], bboxB[2])
	let yB = Math.min(bboxA[3], bboxB[3])
	
	let interArea = (xB - xA) * (yB - yA)

	let iou = 0;
	if (interArea > 0) {
		let bboxAArea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
		let bboxBArea = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
		
		iou = interArea / (bboxAArea + bboxBArea - interArea);
	}
	return iou;
}

function matrixIntersection(image1, image2) {
	// https://www.geeksforgeeks.org/find-the-intersection-of-two-matrices/
	// https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Pixel_manipulation_with_canvas
	
	// if (image1.width !== image2.width && image1.height !== image2.height) {
	// 	alert('she be broke, we be sad widths:' +  image1.width + " " + image2.width + " heights:" +  image1.height + " " + image2.height);
	// 	return;
	// }

	let interset = [];

	let column = 0;
	let row = 0;
	while (column < image1.width) {
		while (row < image1.height) {
			let rc1 = image1.data[((row * (image1.width * 4)) + (column * 4))];
			let gc1 = image1.data[((row * (image1.width * 4)) + (column * 4)) + 1];
			let bc1 = image1.data[((row * (image1.width * 4)) + (column * 4)) + 2];

			let rc2 = image2.data[((row * (image2.width * 4)) + (column * 4))];
			let gc2 = image2.data[((row * (image2.width * 4)) + (column * 4)) + 1];
			let bc2 = image2.data[((row * (image2.width * 4)) + (column * 4)) + 2];

			if (rc1 === rc2) {
				interset.push(rc1);
			} else {
				interset.push('*');
			}

			if (gc1 === gc2) {
				interset.push(gc1);
			} else {
				interset.push('*');
			}

			if (bc1 === bc2) {
				interset.push(bc1)
			} else {
				interset.push('*');
			}

			row += 1;
		}
		column += 1;
	}

	return interset;
}
