---
inclusion: auto
---

# Domain Context: Food Delivery Fraud Detection

## Problem Domain

### The Fraud Pattern
In food delivery, some customers manipulate images to get refunds:
- Add fake mold/burns/defects using photo editing
- Use AI tools (DALL-E, Midjourney, etc.) to generate realistic defects
- Submit manipulated image claiming food was bad
- Get refund from platform (often automatic)

### Business Impact
- Restaurants lose money on legitimate orders
- Delivery platforms often side with customer by default
- Manual verification is subjective and time-consuming
- No technical tools available for restaurants

### Why It's Hard
- AI-generated defects look increasingly realistic
- Legitimate complaints exist (real bad food)
- Need to distinguish: real defect vs digital manipulation
- False accusations damage customer relationships

## Solution Approach

### Conservative Strategy
**Key Principle**: Better to miss fraud than falsely accuse
- Optimize for **precision** over recall
- High confidence threshold for "fraud" classification
- Uncertain cases → manual review, not automatic accusation
- Goal: reduce obvious fraud, not catch everything

### Why Conservative?
- **False Positive**: Restaurant loses customer trust, may stop using service
- **False Negative**: Fraud goes undetected, restaurant loses money
- **Trade-off**: Missing some fraud is acceptable, false accusations are not

### Model Role
This model is ONE component in decision pipeline:
1. Restaurant uploads suspicious image
2. Model analyzes → confidence score + indicators
3. LLM explains findings in human terms
4. Restaurant makes final decision (not automatic)

## Dataset Context

### Training Data
- **Originali**: Real food photos (buono=good, cattivo=naturally bad)
- **Modificate**: Synthetically manipulated (various generators)
- **Metadata**: food_category, defect_type, generator

### Reality Gap
- Training: synthetic modifications (controlled)
- Production: real fraud (unknown techniques)
- Mitigation: continuous monitoring + retraining

### Key Categories
- **Food types**: pizza, carne, sushi, etc.
- **Defect types**: bruciato, crudo, marcio, ammuffito, insetti
- **Generators**: gpt_image_1_5, gpt_image_1_mini (training only)

## Evaluation Priorities

### Primary Metrics
1. **Precision** - Most important (avoid false positives)
2. **PR-AUC** - Better than ROC-AUC for imbalanced data
3. **F1** - Balance, but precision weighted higher

### Secondary Analysis
- **Group metrics**: Which food/defect types are harder?
- **False positives**: What legitimate images trigger detection?
- **Confidence distribution**: Are predictions well-calibrated?

### Threshold Selection
- Training default: 0.5
- Production likely: 0.7-0.8 (conservative)
- May vary by food category if needed

## Production Deployment

### User Flow
1. Restaurant receives complaint with image
2. Uploads to Verifoto web app
3. Gets analysis report:
   - Confidence score (0-100%)
   - Technical indicators found
   - Human-readable explanation
   - Recommendation (likely fraud / uncertain / likely legitimate)
4. Restaurant decides whether to dispute

### Model Requirements
- Fast inference (<1s per image)
- Works on any food image (no metadata needed)
- Confidence scores (not just binary)
- Explainable (which features triggered detection)

### Success Criteria
- Reduce fraudulent refunds by X%
- Maintain restaurant trust (low false positive rate)
- Easy to use (non-technical users)
- Objective evidence for disputes

## Ethical Considerations

### Not Accusatory
- Tool provides evidence, not verdict
- Restaurant makes final decision
- Customers not directly accused
- Used for internal decision-making

### Transparency
- Explain why image flagged
- Show technical indicators
- Acknowledge uncertainty
- Avoid absolute claims

### Fairness
- No bias against legitimate complaints
- Conservative threshold protects customers
- Manual review for borderline cases
- Continuous monitoring for bias

## Future Directions

### Model Improvements
- Better generalization to real fraud
- Explainability (attention maps, feature importance)
- Multi-modal (image + metadata if available)
- Ensemble methods

### Data Strategy
- Collect real fraud cases (with permission)
- Expand food categories
- Test against new AI generators
- Adversarial testing

### Product Evolution
- Integration with delivery platforms
- Automated dispute filing
- Trend analysis (fraud patterns over time)
- Restaurant dashboard
