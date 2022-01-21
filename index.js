const express = require('express')
const multer = require('multer')
const jpeg = require('jpeg-js')

const tf = require('@tensorflow/tfjs-node')
const nsfw = require('nsfwjs')
const { cos } = require('@tensorflow/tfjs-node')

const app = express()
const upload = multer()

let _model

const convert = async(img) => {
    // Decoded image in UInt8 Byte array
    const image = await jpeg.decode(img, true)

    const numChannels = 3
    const numPixels = image.width * image.height
    const values = new Int32Array(numPixels * numChannels)

    for (let i = 0; i < numPixels; i++)
        for (let c = 0; c < numChannels; ++c)
            values[i * numChannels + c] = image.data[i * 4 + c]

    return tf.tensor3d(values, [image.height, image.width, numChannels], 'int32')
}



app.post('/nudity-check', upload.single('image'), async(req, res) => {
    let label;
    if (!req.file) res.status(400).send('Missing image multipart/form-data')
    else {
        const image = await convert(req.file.buffer)
        const predictions = await _model.classify(image);
        image.dispose();
        let max = Math.max.apply(Math, predictions.map(function(label) { return label.probability; }));
        // console.log(max);
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i].probability == max) {
                label = (predictions[i].className);
                break;
            }
        }
        if (label == 'Sexy' || label == 'Porn') {
            return res.json({ msg: 'this picture contains nudity which is against our policy so we do not allow to upload it' });
        }
        res.json({ predictions: predictions, picture_details: { name: label, probability: max } });
    }
})



const load_model = async() => {
    _model = await nsfw.load()
}

// Keep the model in memory, make sure it's loaded only once
load_model().then(() => app.listen(8080, console.log('app is running')))