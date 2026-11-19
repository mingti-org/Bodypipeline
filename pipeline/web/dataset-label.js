(function (root, factory) {
  const datasetLabel = factory();
  if (typeof module === "object" && module.exports) module.exports = datasetLabel;
  else root.datasetLabel = datasetLabel;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  return (row) =>
    [row?.dataset_name || row?.instruction, row?.batch_name]
      .filter(Boolean)
      .join(" / ");
});
