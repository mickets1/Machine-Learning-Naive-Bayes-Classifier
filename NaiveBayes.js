/* eslint-disable jsdoc/require-jsdoc */

/**
 * Naive Bayes Algorithm.
 */
class NaiveBayes {
  constructor () {
    this.summaries = null
  }

  /**
   * Fits the model by separating the dataset and summarizing it.
   *
   * @param {number[]} x - Input data.
   * @param {number[]} y - Class labels
   * @returns {object[]} - The predictions.
   */
  fit (x, y) {
    const separated = this.#separateByClass(x, y)
    const summaries = this.#summarizeDataset(separated)
    this.summaries = summaries
    return summaries
  }

  /**
   * Predicts the best label based on probabilities.
   *
   * @param {object[]} x - The datapoints.
   * @returns {object[]} - The best predictions.
   */
  predict (x) {
    const predictions = []

    for (const row of x) {
      const predictionObj = {}

      for (const summary of this.summaries) {
        const className = summary.className
        const means = summary.means
        const stdevs = summary.stdevs

        let probability = 0

        for (let i = 0; i < row.length; i++) {
          const mean = means[i]
          const stdev = stdevs[i]
          const value = row[i]

          // Multiplying probabilities to avoid numerical underflow.
          probability += Math.log(this.#calculateProbability(value, mean, stdev))
        }

        predictionObj[className] = probability
      }

      predictions.push(predictionObj)
    }

    return this.#calculateBestPrediction(predictions)
  }

  /**
   * Separates the dataset by class labels.
   *
   * @private
   * @param {number[]} x - Input data.
   * @param {number[]} y - Class labels
   * @returns {object} - Separated dataset.
   */
  #separateByClass (x, y) {
    const separated = {}

    for (let i = 0; i < y.length; i++) {
      const label = y[i]

      if (!separated[label]) {
        separated[label] = []
      }

      separated[label].push(x[i])
    }

    return separated
  }

  /**
   * Summarizes the dataset by calculating mean, standard deviation.
   *
   * @private
   * @param {object} separatedDataset - Separated dataset.
   * @returns {object[]} - Dataset summaries.
   */
  #summarizeDataset (separatedDataset) {
    const summaries = []

    for (const className in separatedDataset) {
      const classData = separatedDataset[className]

      // Divide means and stdevs into respective class name.
      const classSummary = {
        className,
        means: [],
        stdevs: []
      }

      const rowToColumns = this.#convertRowToColumn(classData)

      for (let i = 0; i < rowToColumns.length; i++) {
        // Column data for each class
        const columnData = rowToColumns[i]
        const meanValue = this.#mean(columnData)
        const stdevValue = this.#stdev(columnData, meanValue)

        classSummary.means.push(meanValue)
        classSummary.stdevs.push(stdevValue)
      }
      summaries.push(classSummary)
    }

    return summaries
  }

  /**
   * Converts the rows to columns to be able to calculate mean and stdev.
   *
   * @param {Array} classData Datapoints
   * @returns {Array} rows to columns.
   */
  #convertRowToColumn (classData) {
    const rowToColumnData = []

    for (let i = 0; i < classData[0].length; i++) {
      const tempData = []

      for (const row of classData) {
        tempData.push(row[i])
      }

      rowToColumnData.push(tempData)
    }

    return rowToColumnData
  }

  // Calculates the mean.
  #mean (classificationData) {
    let sum = 0.0

    for (const number of classificationData) {
      sum += number
    }

    return sum / classificationData.length
  }

  // Calculates the standard deviation.
  #stdev (classificationData, mean) {
    const avg = mean
    let variance = 0.0

    for (const number of classificationData) {
      variance += Math.pow(number - avg, 2)
    }

    return Math.sqrt(variance / (classificationData.length - 1))
  }

  // Calculate the Gaussian probability distribution function for x
  #calculateProbability (x, mean, stdev) {
    const exponent = Math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (Math.sqrt(2 * Math.PI) * stdev)) * exponent
  }

  /**
   * Calculates the best prediction from the summary.
   *
   * @private
   * @param {object[]} predictions predictions for each class
   * @returns {object[]} - The best prediction.
   */
  #calculateBestPrediction (predictions) {
    const bestPredictions = []

    for (const prediction of predictions) {
      let bestLabel = null
      let bestProb = null

      for (const className in prediction) {
        const probability = prediction[className]

        if (!bestLabel || probability > bestProb) {
          bestProb = probability
          bestLabel = className
        }
      }

      bestPredictions.push({ label: bestLabel, probability: bestProb })
    }

    return bestPredictions
  }

  /**
   * Calculates the accuracy score for a set of predictions against test dataset.
   *
   * @param {object[]} preds - predictions.
   * @param {object[]} y - test dataset labels.
   * @returns {string} - accuracy score.
   */
  accuracyScore (preds, y) {
    let count = 0

    for (let i = 0; i < preds.length; i++) {
      const predictedLabel = preds[i].label
      const actualLabel = y[i]

      if (parseInt(predictedLabel) === actualLabel) {
        count++
      }
    }

    const accuracy = `Accuracy: ${(count / y.length * 100).toFixed(2)}%`
    const classified = ` - ${count}/${y.length} correctly classified`

    return accuracy + classified
  }
}

export default NaiveBayes
