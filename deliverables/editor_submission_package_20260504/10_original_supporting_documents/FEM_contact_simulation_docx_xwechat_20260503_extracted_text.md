# Extracted Text

Source: `C:\Users\SWH\Documents\xwechat_files\wxid_ty6107lq8k9q22_6147\msg\file\2026-05\肺结节接触仿真.docx`

This is a plain-text extraction for manuscript planning; inspect the original DOCX for formatting and figures.

一、仿真结果

1.有无肺结节

左图为无结节，右图为有2.5mm直径的结节（深度均为0mm），右边的图例是指的变形（长度）

2.有结节，但大小有差异

从左往右分别是直径为2.5mm、5mm、7.5mm的结节（深度均为0mm）

3.有结节，大小为5mm，深度有差异

从左往右分别是深度为2mm、5mm、7mm的结节（直径均为5mm）

4.有结节，大小有差异为2.5mm和5mm，深度无差异为2mm

从左往右分别是大小2.5mm和5mm的结节（深度均为2mm）

二、仿真过程说明

1.仿真模型

利用ansys中的space claim进行模型建立，肺结节采用球形，直径不断变化，2.5mm、5mm、7.5mm；压力传感器采用长方体，并根据圣维南原理将压力传感器的尺寸在仿真过程中进行放大，尺寸为45mm*50mm*1mm，以减少边界条件对仿真结果的影响；肺组织隔层采用长方体，横截面尺寸为45mm*50mm，厚度不断变化，2mm、5mm、7mm，在深度变化仿真中并与压力传感器相互交错，未完全重合，以便施加接触条件。

2.仿真过程

利用ansys中的静态结构（static structural）仿真进行仿真模拟，将模型导入mechanical中进行仿真，通过设置结构几何体的材料属性，定义接触关系（接触类型为绑定，其中有深度的，肺组织和压力传感器之间为无分离），划分网格（网格尺寸均为0.003m），定义边界条件和加载状况，仿真求解一系列流程得到模型的仿真结果。在边界条件的设定中，由于模型必须进行固定，因而在压力传感器和肺组织的两个相对的面设置固定位移边界条件；在加载状态的设定中，根据力的相互作用与反作用的原理，通过对肺结节施加25N大小的力（方向垂直于压力传感器）进行模拟。

3.仿真结果的说明

仿真结果均是总变形量，由于材料均为各向同性材料且均匀，因而可以使用总变形量来间接反映力大小的分布情况。

附：（可以不写，数值有点不严谨）

材料属性的定义：

压力传感器：塑胶

参考文献：无（材料种类比较多，找不到特定的参数，所以让ai生成了一个，结果也比较明显）

密度：1150kg/m3

弹性模量：400MPa

泊松比：0.4

肺结节：结构钢

参考文献：无（没找到说明肺结节材料参数的文献，所以用的比较硬的“钢”的材料参数）

密度：7850kg/m3

弹性模量：200000MPa

泊松比：0.33

肺组织：肺

参考文献：Bou Jawde S, Takahashi A, Bates J H T, et al. An analytical model for estimating alveolar wall elastic moduli from lung tissue uniaxial stress-strain curves[J]. Frontiers in physiology, 2020, 11: 121.

密度：400kg/m3

弹性模量：3KPa

泊松比：0.45

手部压力：25N

参考文献：Aldien Y, Welcome D, Rakheja S, et al. Contact pressure distribution at hand–handle interface: role of hand forces and handle size[J]. International Journal of Industrial Ergonomics, 2005, 35(3): 267-286.
