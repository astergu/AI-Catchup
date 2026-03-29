# Deep Dive: Visual Search (Pinterest "Lens" / Google Photos)

*Deep dive into Question #8. A strong candidate understands embedding extraction, Approximate Nearest Neighbor (ANN) search at scale, and the latency constraints of mobile deployment.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Input type:** User-uploaded photo? Cropped sub-region of an image? Screenshot?
- **Scale:** How many images in the index? (Pinterest has ~10 billion images)
- **Latency:** < 500ms end-to-end including mobile upload time?
- **Success Metric:** Recall@K (do any of the Top K results match what the user wanted)?

### 2. Architecture Overview

```
[User uploads image on mobile]
          |
  [Image Preprocessing]  → resize, normalize
          |
  [Feature Extraction]   → CNN / ViT → embedding vector (e.g., 512-dim)
          |
  [ANN Search Service]   → FAISS / ScaNN → Top K similar embeddings
          |
  [Re-ranking (optional)] → Cross-modal or fine-grained model
          |
  [Return similar images / products]
```

### 3. Feature Extraction

The core of visual search is converting images into a compact embedding vector that captures semantic similarity.

**Model choices:**
| Model | Strength | Typical Use |
|---|---|---|
| **ResNet / EfficientNet** | Fast inference, proven on visual similarity | Product image search |
| **ViT (Vision Transformer)** | Higher accuracy, better global context | High-quality visual search |
| **CLIP (Contrastive Language-Image Pretraining)** | Cross-modal: image and text in the same embedding space | "Search by image OR text description" |

**Training approach:**
- **Metric learning with contrastive loss or triplet loss:** Train the model so that similar images are close in embedding space and dissimilar images are far apart.
- **Anchor:** Query image. **Positive:** Similar image (same product/category). **Negative:** Dissimilar image.
- This is called **Siamese network** training.

**Offline pre-computation:** All indexed images have their embeddings pre-computed and stored. Only the query image embedding needs to be computed at runtime.

### 4. Approximate Nearest Neighbor (ANN) Search

Exact nearest neighbor search over 10 billion 512-dim vectors is too slow (O(N × D) per query). Use ANN:

| Method | How | Library |
|---|---|---|
| **HNSW (Hierarchical Navigable Small World)** | Graph-based index; very fast query, high recall | FAISS, hnswlib |
| **IVF (Inverted File Index)** | Clusters embeddings; search only nearby clusters | FAISS |
| **ScaNN (Scalable Nearest Neighbors)** | Google's production ANN system, optimized for both speed and recall | ScaNN |
| **Product Quantization (PQ)** | Compress vectors to reduce memory footprint | Combined with IVF in FAISS |

**The recall vs. latency trade-off:** More clusters searched = higher recall but higher latency. Tune at serving time.

### 5. Handling Mobile Constraints

This is the key "pro" follow-up. Mobile devices have:
- Limited bandwidth (especially on 4G/low-signal)
- Limited CPU/GPU for local processing
- Variable network latency

**Strategies:**
- **Compress the query image before upload:** Resize to 224×224, JPEG compress at 70% quality. Most visual similarity is preserved; upload size drops 10x.
- **On-device embedding extraction:** Run a small, quantized (INT8) model on the device to produce the embedding locally. Only send the 512-float embedding (2KB) to the server instead of the full image (500KB–5MB). This eliminates most of the upload bandwidth problem.
- **Progressive results:** Return a low-quality first batch immediately, then refine in the background.
- **Caching:** Cache popular query embeddings and their results at the edge (CDN).

### 6. Sub-image / Object Search

Pinterest "Lens" allows searching within a cropped region of an image. This requires:
- **Object detection (YOLO / Faster R-CNN):** Detect objects within the image first.
- **Region-level embedding:** Extract features from the detected region's bounding box, not the full image.
- **Multi-scale features:** Objects at different scales should produce similar embeddings.

### 7. Evaluation

| Metric | Description |
|---|---|
| **Recall@K** | Are any of the Top K results genuinely similar? K is typically 10 or 50. |
| **Precision@K** | What fraction of Top K results are relevant? |
| **Mean Average Precision (mAP)** | Aggregate quality of the ranked list. |
| **Latency P95** | 95th percentile end-to-end query latency. |

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Embedding** | Pixel-level comparison (MSE). | CNN/ViT embeddings with contrastive training. | Discusses CLIP for cross-modal search and metric learning trade-offs. |
| **Retrieval** | Exact KNN. | Mentions ANN search. | Compares HNSW vs. IVF, discusses recall/latency trade-offs and index update strategies. |
| **Mobile** | Ignores mobile constraints. | Mentions image compression before upload. | Proposes on-device embedding extraction to send only 2KB to the server. |
| **Index updates** | Does not address. | Mentions re-indexing new images. | Discusses incremental indexing (append new vectors without rebuilding the full index) and handling deleted images. |

---

## How to "Fail" a Candidate (Red Flags)

- **Exact nearest neighbor search:** O(N) over 10B images is computationally infeasible at query time. Any solution without ANN fails on scale.
- **No embedding training discussion:** Simply using a classification model's features (not trained for similarity) produces suboptimal embeddings for retrieval.
- **Ignoring the update problem:** New images are added constantly. The ANN index must support incremental updates without full rebuilds every time.
