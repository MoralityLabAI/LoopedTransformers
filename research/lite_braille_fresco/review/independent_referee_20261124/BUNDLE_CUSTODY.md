# Review publication and custody

These two documents are byte-identical to the corresponding Git blobs in the independent review bundle supplied by the maintainer. The review's original signed commit is `8102fb2190e3c0d30a5a7fad93694d4eaa0af6bd`, parent `3d8243818858c19c5031c36e2bf7e328f831cf25`. Bundle SHA-256: `f0fd04744b2199e2d76647408fdc097e42e7ebed1e0cd38825ae9b80c1cda1f6`.

This response branch publishes the documents only; relative links into `audit/` will resolve when the original independent branch is imported. The original signed commit and all audit scripts and outputs should be published from the supplied bundle using a full local clone. Keep the original independent review branch intact. Do not mistake this copy commit for the original signed commit.

```sh
git fetch origin main
git fetch /path/to/lbf-independent-referee-review-20261124.bundle research/lbf-independent-referee-review-20261124:research/lbf-independent-referee-review-20261124
git push origin research/lbf-independent-referee-review-20261124
```
