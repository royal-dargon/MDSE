# MDSE

This is the repository of *MODALITY-DEPENDENT SENTIMENTS EXPLORING FOR MULTI-MODAL SENTIMENT CLASSIFICATION*

* **Overview of the proposed MDSE framework**

<img src="./img/mdse_new.png" style="zoom:10%;" />

## MDSE on Multimodal Sentiment Classification Results

* MVSA-S datasets

|       Model       |    ACC    |   W-F1    |
| :---------------: | :-------: | :-------: |
|   MultiSentiNet   |   69.84   |   69.63   |
|     Co-MN-Hop     |   70.51   |   70.01   |
|        MFF        |   71.44   |   71.06   |
|       MGNNS       |   73.77   |   72.70   |
|       TBNMD       |   75.22   |   73.46   |
|       CLMLF       |   75.33   |   73.46   |
| MDSE(Base/VGG-19) |   74.33   |   74.38   |
|   MDSE(VGG-19)    |   74.22   |   73.20   |
| MDSE(Base/ResNet) |   73.33   |   72.74   |
|   MDSE(ResNet)    |   75.93   |   74.83   |
|  MDSE(Base/VIT)   |   73.77   |   72.52   |
|    MDSE(ours)     | **76.22** | **75.71** |

* MVSA-M datasets

|   Model(MVSA-M)   |    ACC    |   W-F1    |
| :---------------: | :-------: | :-------: |
|   MultiSentiNet   |   68.86   |   68.11   |
|     Co-MN-Hop     |   68.92   |   68.83   |
|        MFF        |   69.62   |   69.35   |
|       MGNNS       |   72.49   |   69.34   |
|       TBNMD       |   70.72   |   67.94   |
|       CLMLF       |   72.00   |   69.83   |
| MDSE(Base/VGG-19) |   70.17   |   68.74   |
|   MDSE(VGG-19)    |   71.23   |   68.10   |
| MDSE(Base/ResNet) |   70.29   |   67.65   |
|   MDSE(ResNet)    |   71.97   |   69.92   |
|  MDSE(Base/VIT)   |   71.00   |   67.83   |
|    MDSE(ours)     | **72.31** | **70.12** |



* TWITTER-15 datasets 

| Model(TWITTER-15) |    ACC    |   M-F1    |
| :---------------: | :-------: | :-------: |
|      TomBERT      |   77.15   |   71.75   |
|     CapTrBERT     |   78.01   |   73.25   |
|       TBNMD       |   76.73   |   71.19   |
|       CLMLF       |   78.11   |   74.60   |
| MDSE(Base/VGG-19) |   73.44   |   73.14   |
|   MDSE(VGG-19)    |   75.77   |   73.22   |
| MDSE(Base/ResNet) |   75.04   |   73.75   |
|   MDSE(ResNet)    |   78.05   |   74.92   |
|  MDSE(Base/VIT)   |   75.60   |   74.28   |
|    MDSE(ours)     | **78.49** | **75.29** |

* TWITTER-17 datasets

|       Model       |    ACC    |   M-F1    |
| :---------------: | :-------: | :-------: |
|      TomBERT      |   70.50   |   68.04   |
|     CapTrBERT     |   72.30   |   70.20   |
|       TBNMD       |   71.52   |   70.18   |
|       CLMLF       |   70.98   |   69.13   |
| MDSE(Base/VGG-19) |   70.58   |   70.52   |
|   MDSE(VGG-19)    |   70.98   |   71.02   |
| MDSE(Base/ResNet) |   70.91   |   70.92   |
|   MDSE(ResNet)    |   71.93   |   71.88   |
|  MDSE(Base/VIT)   |   70.98   |   70.91   |
|    MDSE(ours)     | **72.77** | **72.63** |

## Visual Example

*  **Visualizations of sentiment classification scores**

<img src="./img/visual_new.png" style="zoom:25%;" />

Above is a visual analysis of the effect of the MDSE model on private features learning(**PFL**). In the first image-text data pair, the labels of image and text are neutral and positive, respectively, and the final label is positive. From the result, if the text private feature learning(**tPFL**) module is ignored, the final result will be wrong, indicating that the part marked by the green box in the text data is important to the result. In the second data pair, the image part is marked with a green box as the private feature of the image, and it can be found that if ignored, the result will become neutral.

