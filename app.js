import { parseDataset } from './csvparser.js'
import NaiveBayes from './NaiveBayes.js'
import readline from 'readline'

/**
 * Read user input.
 */
function readInput () {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  })

  rl.question('1. Iris Dataset\n2. Banknote Dataset\n: ', (selection) => {
    let dataset = null
    let testDataset = null

    if (parseInt(selection) === 1) {
      dataset = parseDataset('./datasets/iris.csv')
      testDataset = dataset
    } else {
      dataset = parseDataset('./datasets/banknote_authentication.csv')
      testDataset = dataset
    }

    const naiveBayes = new NaiveBayes()

    // y = labels, x = data points.
    const x = dataset[1].x
    const y = dataset[0].y
    naiveBayes.fit(x, y)
    const predictions = naiveBayes.predict(x)

    const acc = naiveBayes.accuracyScore(predictions, testDataset[0].y)
    console.log(acc)
    rl.close()
  })

  rl.on('close', () => {
    process.exit(0)
  })
}

readInput()
