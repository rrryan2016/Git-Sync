Baby Can You See it in my laptop??

From Lab!!

This is motherfxxk from my own laptop!!! 

* Key words 
> 爬虫 数据清洗
* 建立数据库的步骤
> 数据预处理 - 建立数据库连接 - 打开数据连接 - 建立数据库命令 - 运行数据库命令 - 保存数据库命令 - 关闭数据库连接
* TCP/IP的四层结构
> 应用层 传输层 网络互连层 主机到网络层  
* 什么是MVC结构 ji简要介绍各层的作用
> Model View Control 
* 数据如何清洗，怎么处理缺失值？
> 分析数据的统计信息、分布情况、缺失情况来定。在数据质量较好的前提下尽可能保留更多的数据。缺失值的处理方法较多，也是根据具体特征和业务来定，可以随机填充、均值填充、或采用简单算法如KNN，聚类进行填充。如果某些特征或某些样本的缺失率太大，可以考虑直接舍弃，视具体情况而定。
* 数据挖掘工程师
> 具体工作的可能：建模 数据分析 ETL工作
* 数据分析工程师
> 自己对数据有充分的了解
> 准备R语言还有数据分析统计和可视化角度入手
* ETL工作
> 做数据的前期，包括数据清洗 整理 校验 从SQL语言入手
* SQL中inner join 和outer join的区别？
> 
* 如果写的程序跑的非常慢，多方面分析这个问题？
> 检查程序是否有多层嵌套循环，优化
> 检查程序是否有很耗时的操作，看能否优化为多线程并行执行
> 检查数据量是否非常大，考虑是否可以用分布式计算模型。
> 
* 写算法
> 输入根节点和两个子节点，找最小公共父节点（二叉树只有孩子节点）
> 写一个栈，添加接口，返回当前栈最小值
> 单链表如何判断有环，
> 从大数据中找出topk
> 简历信息！！
> 在一堆单词里面找出现次数最多的k个
> hadoop，数据结构与算法
> hadoop原理，shuffle如何排序，map如何切割数据，如何处理数据倾斜，join的mr代码如何写
> 动态规划，树结构，链表结构等等
> 把一个完整的数据挖掘流程讲一下，从预处理，特征工程，到模型融合。
> 介绍常用的算法，gbdt和xgboost区别，具体怎么做预处理，特征工程，模型融合常用方式，融合一定会提升吗
> 如何在海量数据中查找给定部分数据最相似的top200向量，向量的维度也很高（直接就说可以用KD树，聚类，hash）
> 
> 简历里面写了机器学习：所以要准备这个：
> 为什么LR需要归一化或者取对数，为什么LR把特征离散化后效果更好
> 为什么把特征组合之后还能提升，反正这些基本都是增强了特征的表达能力，或者说更容易线性可分
> 经典算法的推导 原理
> 各个损失函数之间的区别 使用场景
> 如何并行化
> 有什么关键参数
> 特征选择方法有哪些(能说出来10种以上加分)
> 如何克服过拟合，欠拟合（决策树剪枝 L1 L2正则）
> L0，L1，L2正则化(如果能推导绝对是加分项，一般人最多能画个等高线，L0是NP问题)
> 上面的这些问题基本都能在《李航：统计学习方法》《周志华：机器学习》里面找到，能翻个4，5遍基本就无压力了
> 没事审视下自己的简历，不要把自己不熟悉的东西写上去，像什么精通之类的建议改成了解吧
> 
> 
> 
* 面试官问你还有问题要问没
> 比如问点现在这个部门做的业务，遇到过的问题，部门发展的一个规划
> 

剑指offer
https://blog.csdn.net/mmc_maodun/column/info/mmc-offer
微软面试100题
* LR为什么用sigmoid函数。这个函数有什么优点和缺点？为什么不用其他函数？
* SVM原问题和对偶问题关系？
* KKT条件用哪些，完整描述
* 有一堆已经分好的词，如何去发现新的词？（用这个词和左右词的关系。互信息 新词的左右比较丰富，有的老词的左右也比较丰富。还要区分出新词和老词。）
* L1正则为什么可以把系数压缩成0，坐标下降法的具体实现细节
* spark原理
* 说一下进程和线程 再就说之间的区别
* 线程安全的理解
* 有哪些线程安全的函数
* 数据库中主键、索引和外键。以及作用 一个表可以没有主键，可以有索引
* 
* 准备简历的问题
> 不要装逼写2页，我很多项目比赛都没写进去，只写了几个名次靠前的比赛，能吹一点的项目，其他没写进去的可以找机会主动说出来,项目即使很水，也要吹的很难很厉害的样子
> 
> 
> 
> 
* 对于机器学习你都学了哪些？讲一个印象深的
* SVM怎么防止过拟合
* 决策树如何防止过拟合（剪枝 前剪枝 后剪枝 REP剪枝）
* 为什么要把原问题转换为对偶问题？
> 因为原问题是凸二次规划问题，转换为对偶问题更加高效。
* 为什么L1正则可以实现参数稀疏，而L2正则不可以？
> L1正则因为是绝对值形式，很多系数被压缩为0,。而L2正则是很多系数被压迫到接近于0，而不是0
* 为什么L1很多系数可以被压缩为0，L2是被压缩至接近于0？
> 图像上，L1正则是正方形，L2正则是圆形。
> L1正则的往往取到正方形顶点，即有很多参数为0
> L2正则往往去不到圆形和参数线的交点，即很多分量被压缩到接近于0
* 实写代码
* TODO
> https://blog.csdn.net/wuxiaosi808/article/details/77374939
> 从二 3 开始看
> 
> 微软100题
> https://blog.csdn.net/v_july_v/article/details/6870251
> 
> Final 大招
> https://blog.csdn.net/v_july_v/article/details/6543438
> 
> 算法知识总结：
> http://www.cnblogs.com/tornadomeet/p/3395593.html
> 
> 
> 





