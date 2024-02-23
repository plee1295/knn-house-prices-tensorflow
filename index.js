require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')

function knn (features, labels, predictionPoint, k) {
  // Standardization
  const { mean, variance } = tf.moments(features, 0)
  const scaledPrediction = predictionPoint.sub(mean).div(variance.sqrt())

  return features
    .sub(mean) // Standardization
    .div(variance.sqrt()) // Standardization
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .sqrt()
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
    .slice(0, k)
    .reduce((acc,pair) => acc + pair.get(1), 0) / k
}

let { features, labels, testFeatures, testLabels } = loadCSV('./kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living', 'yr_built'],
  labelColumns: ['price']
})

features = tf.tensor(features)
labels = tf.tensor(labels)

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10)
  const err = (testLabels[i][0] - result) / testLabels[i][0] * 100
  console.log('Error', err, '%')
})

// Normalize vs standardize calculator ?
// Normalized (0-1): (x - min) / (max - min)
// Standardized: (x - mean) / standard deviation
