import * as tf from '@tensorflow/tfjs';
/**
 * Apply a patch tensor to a 2D tensor at given (x, y) slice start position.
 * 
 * @param {tf.Tensor2D} tensor - The target 2D tensor to modify.
 * @param {tf.Tensor2D} patch - The patch tensor to insert.
 * @param {number} x - Row index where the patch should be inserted.
 * @param {number} y - Column index where the patch should be inserted.
 * @returns {tf.Tensor2D} - The new tensor with the patch applied.
 */
export function applyPatchTo2DTensor(tensor, patch, x, y) {
  return tf.tidy(() => {
    const [H, W] = tensor.shape;
    const [h, w] = patch.shape;

    if (x + h > H || y + w > W) {
      throw new Error("Patch goes out of bounds of the original tensor");
    }

    // Extract parts of the tensor around the patch
    const top = tensor.slice([0, 0], [x, W]);
    const middle = tensor.slice([x, 0], [h, W]);
    const bottom = tensor.slice([x + h, 0], [H - x - h, W]);

    // For the middle rows, replace the middle slice with the patch
    const newMiddle = tf.stack(
      middle.unstack().map((rowTensor, i) => {
        const rowLeft = rowTensor.slice([0], [y]);
        const rowRight = rowTensor.slice([y + w], [W - y - w]);
        return tf.concat([rowLeft, patch.slice([i, 0], [1, w]).squeeze([0]), rowRight]);
      })
    );

    // Combine everything
    return tf.concat([top, newMiddle, bottom], 0);
  });
}

export function makeArray(dx, dy)
{
    const arr = new Array(dx);
    for (let i = 0; i < dx; ++i)
    {
        arr[i] = new Array(dy).fill(0);
    }
    return arr;
}

export async function loadTerrain(fname) {
  const fpath = "../WaterSim/" + fname + ".txt";
  const res = await fetch(fpath);
  const text = await res.text();
  
  const lines = text.trim().split('\n');
  
  let xmin = 9999, ymin = 9999, zmin = 9999;
  let xmax = -9999, ymax = -9999, zmax = -9999;
  
  let parsed_lines = [];
  
  for (const line of lines)
  {
      if (line.startsWith('#'))
      {
          continue;
      }
      const [xStr, yStr, zStr, color] = line.trim().split(/\s+/);
      const x = parseInt(xStr, 10);
      const y = parseInt(yStr, 10);
      const z = parseInt(zStr, 10);
  
      parsed_lines.push([x,y,z,color]);
      
      if (color === '000000'){
          // 000000 determines base size
          xmin = Math.min(xmin, x)
          ymin = Math.min(ymin, y)
          zmin = Math.min(zmin, z)
          xmax = Math.max(xmax, x)
          ymax = Math.max(ymax, y)
          zmax = Math.max(zmax, z)
      }
  }
  
  let M = xmax - xmin + 1 + 2;
  let N = ymax - ymin + 1 + 2;
  
  const terrainData = makeArray(M, N);
  for (const parsed of parsed_lines)
  {
      const [x, y, z, color] = parsed;
      if (color == '8f563b')
      {
          terrainData[x - xmin + 1][y - ymin + 1] = Math.max(terrainData[x - xmin + 1][y - ymin + 1], z - zmax);
      }
  }
  
  const waterData = makeArray(M, N);
  const sourceData = makeArray(M, N);
  for (const parsed of parsed_lines)
  {
      const [x, y, z, color] = parsed;
      if (color == '639bff' || color == 'fbf236')
      {
          waterData[x - xmin + 1][y - ymin + 1] = Math.max(waterData[x - xmin + 1][y - ymin + 1], z - terrainData[x - xmin][y - ymin]);
      }
      if (color == 'fbf236')
      {
          sourceData[x - xmin + 1][y - ymin + 1] += 1
      }
  }

  return [terrainData, waterData, sourceData];
}