import fs from 'fs'

/**
 * Parses the dataset and returns X and y arrays.
 *
 * @param {string} dataset Search path.
 * @returns {Array} An array containing class labels and datapoints.
 */
function parseDataset (dataset) {
  const result = []
  const fileContent = fs.readFileSync(dataset, 'utf-8')
  const rows = fileContent.split('\n')
  // Remove header
  rows.shift()

  const x = []
  const y = []
  const labelSet = new Set()

  for (const row of rows) {
    const dataSplit = row.split(',')

    const xValues = []
    for (let i = 0; i < dataSplit.length - 1; i++) {
      xValues.push(parseFloat(dataSplit[i]))
    }

    const label = dataSplit[dataSplit.length - 1].trim()

    if (!labelSet.has(label)) {
      labelSet.add(label)
    }

    x.push(xValues)
    y.push([...labelSet].indexOf(label))
  }

  result.push({ x })
  result.unshift({ y })

  return result
}

export { parseDataset }
