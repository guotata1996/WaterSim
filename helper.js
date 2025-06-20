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