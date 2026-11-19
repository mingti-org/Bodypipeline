const assert = require("node:assert/strict");
const test = require("node:test");

const datasetLabel = require("../pipeline/web/dataset-label.js");

test("appends the delivery batch to the dataset name", () => {
  assert.equal(
    datasetLabel({
      dataset_name: "sim-sarm-drawer-cycle",
      batch_name: "batch-20261111-000005",
    }),
    "sim-sarm-drawer-cycle / batch-20261111-000005",
  );
});

test("keeps non-delivery labels unchanged", () => {
  assert.equal(datasetLabel({ dataset_name: "regular-dataset" }), "regular-dataset");
});

test("falls back to instruction while preserving the batch", () => {
  assert.equal(
    datasetLabel({ instruction: "open drawer", batch_name: "batch-2" }),
    "open drawer / batch-2",
  );
});
