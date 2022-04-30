# Weighted boxes fusion: Ensembling boxes from different object detection models

- 논문 : [1910.13302.pdf (arxiv.org)](https://arxiv.org/pdf/1910.13302.pdf)

# Abstract + Conclusion

```
Our algorithm (WBF) utilizes confidence scores of all proposed bounding boxes to constructs the averaged boxes.
```

# Related Work

## **Non-maximum suppression (NMS)**

> **원리**
> 

1️⃣ 모든 detection box를 confidence score가 높은 순서대로 정렬 

2️⃣ highest confidence score를 갖는 detection box를 선택 

3️⃣ 이 detection box에 overlap된 다른 box들 제거 

- overlap 기준 → `IoU (intersection-over-union)` > `threshold`

> **NMS의 문제1**    
>     
- hard-coded threshold를 사용해서 overlap되는 box 제거 ⇒ **differentiable model X** 

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166084885-cce6e514-e8eb-43fe-b6e8-22bdd7dd349a.png">

- `M` → 해당 클래스에서 confidence가 가장 높은 bounding box를 의미
- `bi` → 동일한 클래스 내의 bounding box를 의미
- `M`과 `bi`의 IoU가 threshold (`Nt`) 이상이면 0으로 제거 ⇒ differentiable X


> **NMS의 문제2**    
>    

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166084710-20809599-2c59-4462-a6d8-5102b5c6bf76.png">

- Figure 1 처럼 object가 나란히 배열된 경우 → threshold를 설정하기 까다로움
    - (생각) threshold를 0.9 이상으로 설정하지 않는 이상, 제대로 detection한 box 중 일부가 제거될 수 밖에 없음
  
<details>
<summary>📎 NMS  </summary>
<div markdown="1">       
  
1. 모델에서 Box Regression 후 통과한 bounding box들이 이렇게 나오게 된다. bounding box가 많아서 연산량이 많아진다.

![https://blog.kakaocdn.net/dn/c4YJ2J/btqT03hw9df/yyKsBAWQT2hB8a0T2m4smk/img.jpg](https://blog.kakaocdn.net/dn/c4YJ2J/btqT03hw9df/yyKsBAWQT2hB8a0T2m4smk/img.jpg)

2. bounding box를 confidence(사진 속에 적힌 숫자)가 높은 순서대로 정렬한다.

3. 제일 큰 confidence값을 기준으로 하나씩 IoU를 비교하여, 일정 threshold 이상이면 제거한다. 

4. 최종 결과물이 표시 된다.

![https://blog.kakaocdn.net/dn/b0SPNd/btqTKLJ4rED/S1zzslBPr1QjlKHwanAKGK/img.jpg](https://blog.kakaocdn.net/dn/b0SPNd/btqTKLJ4rED/S1zzslBPr1QjlKHwanAKGK/img.jpg)
    

</div>
</details>

    


## **Soft-NMS**

> **원리**
> 
- IoU 정도에 따라 confidence score를 줄이거나 늘림

```
# Soft-NMS
reduces the confidences of the proposals proportional to IoU value 
= lower the confidence scores proportionally to the IoU overlap

# NMS 
completely removing the detection proposals with high IoU and high confidence 
```

> **NMS의 문제1 해결**
> 


<img width="30%" src="https://user-images.githubusercontent.com/90603530/166084799-0feb40bd-dc6c-47cf-8a83-236922e1f3fb.png">

- `M`과 `bi`의 IoU가 threshold 이상일 때 → 0으로 제거하는 대신 confidence score를 감소시킴
- `M`과 `bi`의 IoU가 높으면 높은 가중치를 부여 / IoU가 낮으면 낮은 가중치를 부여
    - 가중치 ↑ → confidence score ↓
- 가우시안 분포를 활용 ⇒ score 연속적 ⇒ differentiable model O

> **NMS의 문제2 해결**
>


<img width="40%" src="https://user-images.githubusercontent.com/90603530/166084819-ddcc254a-526f-4c0e-af02-cfb4b565a938.png">

- `NMS`
    - 해당 클래스에서 **confidence가 가장 높은 bounding box를 선택** + 선택된 bounding box와 IoU가 일정 threshold 이상인 박스들은 모두 제거
        
        ⇒ 동일한 클래스를 지닌 여러 물체가 뭉쳐있는 경우, 하나의 bounding box만을 검출 + 나머지 **bounding box 제거**
        
- `Soft-NMS`
    - bounding box를 (곧바로) 제거하는 대신, **confidence**를 **줄임**
    - 기존의 NMS에서는 제거되었을 bounding box에 Soft-NMS가 적용되면 낮은 confidence로 검출됨 (단, score가 너무 낮은 경우는 box는 제거)
    - (생각) confidence score에 따라 bounding box를 제거하긴 하나, box를 제거하기 전에 보류 기간을 두는 방식

## **Test-time augmentation (TTA)**

> **원리**
> 

1️⃣ 같은 model에 대해 `original image + augmented image`를 사용해서 예측

- augmented image → ex. vertically / horizontally reflected image

2️⃣ 이 예측값에 대한 평균을 계산 

<details>
<summary>📎Ensemble  </summary>
<div markdown="1">       
  
- 어떤 데이터에 대해 여러 모델의 예측값을 평균 내어 → 편향된 데이터를 억제 ⇒ 정확도를 높임
- ex. `TTA` → 이미지 task에서 예측을 할 데이터의 밝기가 어둡거나 밝은 데이터, 객체가 작은 데이터 등과 같이 편향된 데이터가 있을때, **여러 Augmentation 기법을 적용해 평균**을 내게 되면 단일 모델의 output을 예측으로 사용할때 보다 더 높은 성능을 보인다.


</div>
</details>


## The non-maximum weighted (NMW)

> **NMW vs. WBF**
> 

|  | NMW | WBF |
| --- | --- | --- |
| confidence score 값을 바꾸는가 | X | O |
| box에 weight를 주기 위해 IoU 값을 사용하는가 | O | X |
| 원리 | highest confidence score인 box를 선정 + 다른 box 들과 overlap 여부 확인  | 매 단계마다 fused box를 update + 다른 box 들과 overlap 여부 확인  |
| 해당 box를 예측한 model 개수 (N) 사용  | X | O |

## **NMS / Soft-NMS vs. WBF**

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166084919-09dfd1db-4cc4-4d25-9da0-7aba10f95278.png">

> **Example : 모든 model의 box 예측값이 틀린 경우**
> 
- `NMS` **/** `Soft-NMS`
    
    → **simply remove** part of the predictions 
    
    ⇒ 하나의 box만 사용 → (생각) 잘못된 것 中 선택해서 답을 도출 
    
- `WBF`
    
    → used **confidence scores** of **all** proposed bounding boxes to constructs the average boxes 
    
    ⇒ 3개의 box 모두 사용 + 평균 계산 → (생각) ground truth에 더 근접할 가능성 O
    

# Weighted Boxes Fusion

> 원리
> 

0️⃣ 가정 (여기서는 아래의 (1)로 가정하고 진행하나, (2)도 가능)

**(1) 같은 이미지 데이터 + 다른 모델**

(2) 같은 모델 + 다른 이미지 데이터 (ex. original + augmented version) 

1️⃣ single list **B**에 predicted box를 추가 + confidence score **C**가 높은 순서대로 정렬

2️⃣ box cluster를 모아두는 list **L**과 fused box 모아두는 list **F** 정의 

- L의 원소 → **set of boxes** or **single box**
- F의 원소 → **one fused box**

3️⃣ list **F**에 있는 box와 overlap 되는 box를 찾기 위해서, 반복문 진행

- 조건 → `list B의 box와 list F의 box와의 IoU > threshold`
- 실험 결과 → optimal threshold = 0.55

4️⃣ list **F**에 있는 box와 overlap 되는 box를

- 못 찾음 (or list **F**가 비어있음)
    - list **B**에 있던 predicted box를 list **L**과 list **F**의 가장 마지막 원소로 추가 → list **B**의 다음 인덱스로 넘어가서 진행
- 찾음
    - list **B**에 있던 predicted box를 해당 인덱스 (인덱스 이름을 ‘pos’라고 가정) 에 list **L**의 원소로 추가
    - L[pos]에 있는 T개의 box를 활용해서 → F[pos]에 해당하는 box의 confidence score과 coordinate를 다시 계산
        - **Confidence score** for the fused box = **the average confidence** of all boxes
        - **Coordinates** of the fused box = **weighted sums of the coordinates** of the boxes
            - **Weights** = **confidence scores** for the corresponding boxes

<img width="60%" src="https://user-images.githubusercontent.com/90603530/166085012-1f6c5608-f763-4822-a7cd-a695911241d9.png">

5️⃣ list B의 모든 box에 대해 위의 과정을 진행한 후, list F의 confidence score를 다시 계산

- cluster에 들어있는 box의 개수 (T)를 곱하고 + model의 개수 (N)으로 나누기
    - cluster에 들어있는 box의 개수 (T) ↓ = 적은 수의 model만이 예측했다
        
        ⇒ confidence score ↓
        
<img width="30%" src="https://user-images.githubusercontent.com/90603530/166085028-b96c581f-ec8e-4782-9a33-634768fb7e4c.png">

> **code**
> 
- [ensemble_boxes_wbf.py](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py)

```python
overall_boxes = []

for label in Boxes:
    boxes = filtered_boxes[label] # boxes (list) : list B # (1)
		# [label, score, weight, model index, x1, y1, x2, y2]를 원소로 저장

    new_boxes = [] # list L # (2)
    weighted_boxes = [] # list F # (2)

    # Clusterize boxes
    for j in range(0, len(boxes)):
        index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr) 
				# (3) list F에 있는 box 중에서 best IoU인 box 찾아냄

        if index != -1: # (4) list F에 있는 box와 overlap 되는 box 찾음 
            new_boxes[index].append(boxes[j])                 
            weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type) 
        else: #No match found  # (4) list F에 있는 box와 overlap 되는 box 못 찾음 
            new_boxes.append([boxes[j].copy()])
            weighted_boxes.append(boxes[j].copy())

    # Rescale confidence based on number of models and boxes # (5)   
    for i in range(len(new_boxes)):
        weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum() 
				# 각 fused box 마다 confidence score 조정
       
    overall_boxes.append(weighted_boxes)
```

<details>
<summary>📎find_matching_box()  </summary>
<div markdown="1">   
  
 ```python
  def find_matching_box (boxes_list, new_box, match_iou):
      def bb_iou_array(boxes, new_box):
          # bb interesection over union
          xA = np.maximum(boxes[:, 0], new_box[0])
          yA = np.maximum(boxes[:, 1], new_box[1])
          xB = np.minimum(boxes[:, 2], new_box[2])
          yB = np.minimum(boxes[:, 3], new_box[3])

          interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

          # compute the area of both the prediction and ground-truth rectangles
          boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
          boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

          iou = interArea / (boxAArea + boxBArea - interArea)

          return iou

      if boxes_list.shape[0] == 0:
          return -1, match_iou

      # boxes = np.array(boxes_list)
      boxes = boxes_list

      ious = bb_iou_array(boxes[:, 4:], new_box[4:])

      ious[boxes[:, 0] != new_box[0]] = -1

      best_idx = np.argmax(ious) 
      best_iou = ious[best_idx]

      if best_iou <= match_iou:
          best_iou = match_iou
          best_idx = -1

      return best_idx, best_iou
  ```

</div>
</details>

<details>
<summary>📎get_weighted_box()  </summary>
<div markdown="1">       

```python
    def get_weighted_box(boxes, conf_type='avg'):
        """
        Create weighted box for set of boxes
        :param boxes: set of boxes to fuse
        :param conf_type: type of confidence one of 'avg' or 'max'
        :return: weighted box (label, score, weight, model index, x1, y1, x2, y2)
        """
    
        box = np.zeros(8, dtype=np.float32)
        conf = 0
        conf_list = []
        w = 0
        for b in boxes:
            box[4:] += (b[1] * b[4:])
            conf += b[1]
            conf_list.append(b[1])
            w += b[2]
        box[0] = boxes[0][0]
        if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
            box[1] = conf / len(boxes)
        elif conf_type == 'max':
            box[1] = np.array(conf_list).max()
        box[2] = w
        box[3] = -1 # model index field is retained for consistency but is not used.
        box[4:] /= conf
        return bo
``` 
  
</div>
</details>
   


# Datasets

## Open Images Dataset

- 16M bounding boxes
- 600 object classes
- 1.9M images

> **Training set**
> 
- 12.2M bounding boxes
- 500 categories
- 1.7M images (1743042 images)

> **Validation set**
> 
- 41620 images

> **Test set**
> 
- 99999 images

## COCO Dataset

- 200,000 images
- 80 object categories

> **Training set**
> 
- train2017
- 118k labeled images

> **Validation set**
> 
- val2017
- 5k labeled images

> **Test set**
> 
- 20k images

# Evaluation

- the mean average precision (mAP)
    - intersection-over-union (IoU) = 0.5

1️⃣ IoU

- a ratio of overlap between two objects (A and B) to the total area of the two objects combined
    - `A` → a set of **predicted** bounding boxes
    - `B` → a set of **ground truth** bounding boxes

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166085047-a0d47263-a7ef-4322-b366-ad7b6edf42cc.png">
  
2️⃣ Precision

- `t` → threshold value

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166085070-2044a9a4-5fd4-4e81-8bf5-cdb0f060d156.png">
  
3️⃣ final AP

- the average AP over the 500 classes

4️⃣ different IoU threshold 

- The threshold values range from 0.5 to 0.95 with a step size of 0.05
    - (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

5️⃣ AP of single image 

- the mean of the above precision values at each IoU threshold

<img width="30%" src="https://user-images.githubusercontent.com/90603530/166085092-ea81ea0e-c4d7-4a38-a70b-fd084214d8dc.png">
  
# Experiments

## Models

> **Single-shot detector**
> 
- RetinaNet + ResNet
- EfficientDet
- DetectoRS

> **Two-stage detector**
> 
- Faster R-CNN
- Mask R-CNN
- Cascade R-CNN

## An ensemble of two different models
  
<img width="70%" src="https://user-images.githubusercontent.com/90603530/166085123-0a0b7fae-5dac-4b60-a61c-25470e331f61.png">
  
- 다른 모델 + 같은 이미지 데이터
    - EfficientDetB6
    - EfficientDetB7
- grid search → optimal parameter 찾기

## Test-time-augmentation ensemble


<img width="70%" src="https://user-images.githubusercontent.com/90603530/166085138-17ced27f-9484-4da0-8f9b-da22cdef7108.png">
  
- 같은 모델 + 다른 이미지 데이터
    - EfficientDetB7
    - original + augmented(ex. horizontally mirrored) images
- grid search → optimal parameter 찾기

## An ensemble of many different models

### Ensemble of models for COCO Dataset

<img width="70%" src="https://user-images.githubusercontent.com/90603530/166085155-bc0d412d-9520-45df-90e0-9e11553da8fd.png">
  
- individual model보다 ensemble 했을 때의 성능이 더 좋음
- validation set으로 ensemble의 weights와 IoU threshold를 optimize

### Ensemble of RetinaNet models for Open Images Dataset

<img width="50%" src="https://user-images.githubusercontent.com/90603530/166085169-80d04e8a-e275-443c-a503-7b7c6bd629bc.png">
  
- 같은 detector + 다른 backbone
    - RetinaNet single-shot-detector
- grid search → optimal parameter 찾기 위해

### Ensemble of fairly different models for Open Images Dataset

<img width="50%" src="https://user-images.githubusercontent.com/90603530/166085186-5d8e9fab-9a28-4407-8582-3d264a2231ca.png">
  
- previous experiments → WBF method for **similar models**
- current experiments → combining predictions from **highly different models**

# Discussion

### **NMS와 비교 실험**

> **결과**
> 

1️⃣ Raw boxes **(no NMS/WBF)** → `mAP`: 0.1718 

- ∵ **many overlapping boxes (→** FP의 비율 ↑ → precision ↓)

2️⃣ **NMS** 

(1) with **default** IoU threshold = 0.5 (ex. standard model output) → `mAP`: 0.4902

(2) with **optimal** IoU threshold = 0.47 → `mAP`: 0.4906 

- the tiny change from the default threshold

3️⃣ **WBF** with optimal parameters → `mAP`: 0.4532 

(the optimized parameters → IoU threshold = 0.43, skip threshold = 0.21)

> **결과 해석**
> 
- single model에서의 성능 → `WBF` < `NMS`
    
    ∵ **the excessive number** **of** low scored **wrong predictions**  
    
    ⇒ `WBF` works well for **combining boxes for fairly accurate models** 
    
- a large number of overlapping boxes with **different confidence scores** → `WBF` < `NMS`
    - (생각) confidence score의 범위가 크지 않아야 `WBF`를 사용하기에 유리

> **속도**
> 
- `NMS` / `Soft-NMS` 보다 느림
    - 구체적으로, `NMS`보다 3배 느림

# Reference

- [NMS(Non-maximum Suppression) 이란? IOU부터 알자 (tistory.com)](https://mickael-k.tistory.com/147)
- [객체 검출/물체 인식의 NMS보다 좋은 앙상블 방법, Weighted Boxes Fusion(WBF): ensembling boxes for object detection models 논문 리뷰 및 정리 (tistory.com)](https://lv99.tistory.com/74)
- [[논문 읽기] Soft-NMS(2017), Improving Object Detection With One Line of Code (tistory.com)](https://deep-learning-study.tistory.com/606)
- [Test Time Augmentation(TTA) (tistory.com)](https://visionhong.tistory.com/26)
- [Weighted Boxes Fusion — A detailed view | by Sambasivarao. K | Analytics Vidhya | Medium](https://medium.com/analytics-vidhya/weighted-boxes-fusion-86fad2c6be16)
- [GitHub - ZFTurbo/Weighted-Boxes-Fusion: Set of methods to ensemble boxes from different object detection models, including implementation of "Weighted boxes fusion (WBF)" method.](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
- [Improving Object Detection With One Line of Code 리뷰 | by srk lee | Medium](https://medium.com/@lsrock125/improving-object-detection-with-one-line-of-code-%EB%A6%AC%EB%B7%B0-696cfb07c9f6)
- [mean Average Precision(mAP) 계산하기 :: Dead & Street (tistory.com)](https://a292run.tistory.com/entry/mean-Average-PrecisionmAP-%EA%B3%84%EC%82%B0%ED%95%98%EA%B8%B0-1)
