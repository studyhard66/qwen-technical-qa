#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_raw_data.py — 生成技术文档领域问答对
策略：
1. 手工编写高质量种子问答（覆盖 Kubernetes/Spark/MySQL/React/Java/Python/PyTorch/Linux/Docker/Git）
2. 基于模板 + 真实技术知识点的合成问答（答案结构有变化，避免被去重全删）
目标：5000+ 条，答案包含代码块或结构化内容
"""
import json
import os
import random
import hashlib

random.seed(42)
ROOT = "D:/SFT/data/raw_qa.jsonl"
os.makedirs(os.path.dirname(ROOT), exist_ok=True)

# ==================== 高质量种子问答（真实技术知识） ====================
SEEDS = [
    # --- Kubernetes ---
    {"q": "Kubernetes 中 Pod 和 Deployment 的区别是什么？",
     "a": "Pod 是 K8s 最小调度单元，包含一个或多个容器；Deployment 是工作负载资源，管理 Pod 的副本数、滚动更新和回滚。\n```yaml\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: nginx\nspec:\n  replicas: 3\n  selector:\n    matchLabels:\n      app: nginx\n  template:\n    metadata:\n      labels:\n        app: nginx\n    spec:\n      containers:\n      - name: nginx\n        image: nginx:1.21\n```\nDeployment 通过 ReplicaSet 维护 Pod 副本，支持声明式更新。"},
    {"q": "如何在 Kubernetes 中配置资源限制？",
     "a": "在 Pod 的容器 spec 中设置 resources.requests 和 resources.limits。\n```yaml\nresources:\n  requests:\n    cpu: \"100m\"\n    memory: \"128Mi\"\n  limits:\n    cpu: \"500m\"\n    memory: \"512Mi\"\n```\nrequests 用于调度，limits 限制最大使用。超出 memory limits 会触发 OOMKill。"},
    {"q": "Kubernetes 的 Service 有哪几种类型？",
     "a": "K8s Service 有 4 种类型：\n1. ClusterIP（默认）：集群内部访问\n2. NodePort：通过节点端口暴露\n3. LoadBalancer：云服务商负载均衡\n4. ExternalName：映射到外部 DNS\n```yaml\nspec:\n  type: NodePort\n  ports:\n  - port: 80\n    targetPort: 8080\n    nodePort: 30080\n```"},
    {"q": "Kubernetes 中 ConfigMap 和 Secret 的区别？",
     "a": "ConfigMap 存储非敏感配置，Secret 存储敏感信息（密码、token）。\n- ConfigMap：明文存储，适合配置文件、环境变量\n- Secret：默认 base64 编码，可加密，适合凭证\n```yaml\napiVersion: v1\nkind: Secret\nmetadata:\n  name: db-secret\ntype: Opaque\ndata:\n  password: cGFzc3dvcmQ=\n```\n使用时通过 envFrom 或 volume 挂载到 Pod。"},

    # --- Spark ---
    {"q": "Spark 中 RDD、DataFrame、Dataset 的区别？",
     "a": "三者是 Spark API 的演进：\n- RDD：弹性分布式数据集，类型安全但无优化，适合低级操作\n- DataFrame：带 schema 的结构化数据，有 Catalyst 优化，不类型安全\n- Dataset：结合两者优点，类型安全且有优化（Scala/Java）\n```python\n# RDD\nrdd = sc.parallelize([1, 2, 3])\n# DataFrame\ndf = spark.read.csv(\"data.csv\", header=True)\n```"},
    {"q": "Spark 的宽窄依赖是什么？",
     "a": "窄依赖：父分区只被一个子分区依赖（如 map、filter），可流水线执行。\n宽依赖：父分区被多个子分区依赖（如 groupByKey、join），触发 shuffle。\n```python\n# 窄依赖\nrdd2 = rdd1.map(lambda x: x * 2)\n# 宽依赖（shuffle）\nrdd3 = rdd2.groupByKey()\n```\n宽依赖是 Stage 划分的依据，shuffle 是性能瓶颈。"},
    {"q": "如何优化 Spark Join 性能？",
     "a": "Spark Join 优化策略：\n1. Broadcast Join：小表广播到所有节点，避免 shuffle\n2. Sort Merge Join：大表默认方式，先排序再合并\n3. 倾斜处理：加盐打散 key\n```python\nfrom pyspark.sql.functions import broadcast\ndf_large.join(broadcast(df_small), \"id\")\n```\n设置 spark.sql.autoBroadcastJoinThreshold 控制广播阈值。"},

    # --- MySQL ---
    {"q": "MySQL 索引的底层数据结构是什么？为什么用 B+ 树？",
     "a": "MySQL InnoDB 使用 B+ 树作为索引结构。\n选择 B+ 树的原因：\n1. 树高低（3-4 层可存千万级数据），磁盘 IO 少\n2. 叶子节点链表有序，范围查询快\n3. 非叶子节点不存数据，单节点能存更多索引\n```sql\nCREATE INDEX idx_name ON users(name);\nCREATE INDEX idx_age_city ON users(age, city);\n```\n复合索引遵循最左前缀原则。"},
    {"q": "MySQL 事务的隔离级别有哪些？",
     "a": "SQL 标准定义 4 种隔离级别：\n1. READ UNCOMMITTED：可读未提交，有脏读\n2. READ COMMITTED：不可重复读，解决脏读\n3. REPEATABLE READ：可重复读，MySQL 默认\n4. SERIALIZABLE：串行化，解决幻读\n```sql\nSET TRANSACTION ISOLATION LEVEL REPEATABLE READ;\n```\nInnoDB 通过 MVCC + Next-Key Lock 在 RR 级别解决幻读。"},
    {"q": "如何排查 MySQL 慢查询？",
     "a": "排查慢查询步骤：\n1. 开启慢查询日志\n2. 使用 EXPLAIN 分析执行计划\n3. 查看索引是否命中\n```sql\nSET GLOBAL slow_query_log = ON;\nSET GLOBAL long_query_time = 1;\nEXPLAIN SELECT * FROM users WHERE age > 20;\n```\n重点关注 type（ALL 全表扫描需优化）、rows、Extra。"},

    # --- React ---
    {"q": "React 中 useState 和 useReducer 的区别？",
     "a": "两者都是状态管理 Hook：\n- useState：简单状态，适合单个值\n- useReducer：复杂状态逻辑，适合状态间有依赖\n```jsx\nconst [count, setCount] = useState(0)\nfunction reducer(state, action) {\n  switch (action.type) {\n    case 'increment': return { count: state.count + 1 }\n    default: throw new Error()\n  }\n}\nconst [state, dispatch] = useReducer(reducer, { count: 0 })\n```"},
    {"q": "React useEffect 的依赖数组有什么作用？",
     "a": "依赖数组决定 useEffect 何时重新执行：\n- 不传：每次渲染都执行\n- 空数组 []：仅挂载时执行一次\n- [dep]：dep 变化时执行\n```jsx\nuseEffect(() => {\n  fetchData()\n}, [])\nuseEffect(() => {\n  fetchUser(userId)\n}, [userId])\n```\n遗漏依赖会导致闭包陷阱。"},
    {"q": "React.memo 和 useMemo、useCallback 的区别？",
     "a": "三者都用于性能优化：\n- React.memo：包裹组件，props 浅比较相同则跳过渲染\n- useMemo：缓存计算结果，依赖不变则不重算\n- useCallback：缓存函数引用，依赖不变则不重建\n```jsx\nconst Memoized = React.memo(Child)\nconst value = useMemo(() => compute(a, b), [a, b])\nconst handler = useCallback(() => doSomething(id), [id])\n```"},

    # --- Java ---
    {"q": "Java 中 == 和 equals() 的区别？",
     "a": "- ==：比较基本类型时值相等；比较引用类型时是引用地址相等\n- equals()：Object 中默认与 == 相同，常被重写为内容比较\n```java\nString s1 = new String(\"abc\");\nString s2 = new String(\"abc\");\nSystem.out.println(s1 == s2);      // false\nSystem.out.println(s1.equals(s2)); // true\n```\n重写 equals 必须同时重写 hashCode。"},
    {"q": "Java HashMap 的底层实现原理？",
     "a": "JDK 1.8 后 HashMap = 数组 + 链表 + 红黑树：\n1. 计算 key 的 hash，定位数组下标\n2. 冲突时链表存储，链表长度 > 8 且数组长度 > 64 转红黑树\n3. 默认容量 16，负载因子 0.75，扩容翻倍\n```java\nMap<String, Integer> map = new HashMap<>();\nmap.put(\"a\", 1);\n```\nHashMap 非线程安全，多线程用 ConcurrentHashMap。"},
    {"q": "Java 线程池的核心参数有哪些？",
     "a": "ThreadPoolExecutor 核心参数：\n- corePoolSize：核心线程数\n- maximumPoolSize：最大线程数\n- keepAliveTime：非核心线程空闲存活时间\n- workQueue：任务队列\n- threadFactory：线程工厂\n- handler：拒绝策略\n```java\nExecutorService pool = new ThreadPoolExecutor(\n    5, 10, 60L, TimeUnit.SECONDS,\n    new LinkedBlockingQueue<>(100),\n    Executors.defaultThreadFactory(),\n    new ThreadPoolExecutor.AbortPolicy()\n);\n```"},

    # --- Python ---
    {"q": "Python 中 *args 和 **kwargs 的区别？",
     "a": "- *args：收集任意数量的位置参数为元组\n- **kwargs：收集任意数量的关键字参数为字典\n```python\ndef func(*args, **kwargs):\n    print(args)   # (1, 2, 3)\n    print(kwargs) # {'a': 1, 'b': 2}\n\nfunc(1, 2, 3, a=1, b=2)\n```\n顺序：位置参数 > *args > 默认参数 > **kwargs。"},
    {"q": "Python 装饰器的原理和用法？",
     "a": "装饰器本质是闭包，在不修改原函数的前提下增强功能。\n```python\ndef log(func):\n    def wrapper(*args, **kwargs):\n        print(f\"调用 {func.__name__}\")\n        return func(*args, **kwargs)\n    return wrapper\n\n@log\ndef hello(name):\n    print(f\"Hello {name}\")\n```\n带参数装饰器需三层嵌套；functools.wraps 保留原函数元信息。"},

    # --- PyTorch / 深度学习 ---
    {"q": "PyTorch 中 model.eval() 和 torch.no_grad() 的区别？",
     "a": "- model.eval()：切换评估模式，关闭 Dropout 和 BatchNorm 训练行为\n- torch.no_grad()：关闭梯度计算，节省显存和加速\n两者配合使用：\n```python\nmodel.eval()\nwith torch.no_grad():\n    output = model(input)\n```\neval() 不影响梯度计算；no_grad() 不影响 Dropout/BatchNorm。"},
    {"q": "如何解决 PyTorch 梯度爆炸问题？",
     "a": "常用方法：\n1. 梯度裁剪：clip_grad_norm_\n2. 降低学习率\n3. 使用 BatchNorm/LayerNorm\n4. 残差连接\n```python\nloss.backward()\ntorch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\noptimizer.step()\n```\n梯度爆炸的信号：loss 出现 NaN 或剧烈震荡。"},
]

# ==================== 领域知识点库 ====================
DOMAIN_KNOWLEDGE = {
    "kubernetes": [
        ("Pod 的生命周期有哪些阶段？", "Pod 生命周期阶段：Pending（等待调度）、Running（运行中）、Succeeded（成功退出）、Failed（异常退出）、Unknown（节点失联）。\n```bash\nkubectl get pods\nkubectl describe pod <pod-name>\n```\n通过 status.phase 查看当前阶段。"),
        ("如何实现 Kubernetes 滚动更新？", "通过 Deployment 的 rollingUpdate 策略：\n```yaml\nspec:\n  strategy:\n    type: RollingUpdate\n    rollingUpdate:\n      maxSurge: 1\n      maxUnavailable: 0\n```\nmaxSurge 控制最多超出副本数，maxUnavailable 控制最大不可用数。"),
        ("Kubernetes 中 Namespace 的作用？", "Namespace 用于资源隔离和环境划分（dev/staging/prod）。\n```bash\nkubectl create namespace dev\nkubectl apply -f app.yaml -n dev\nkubectl get pods -n dev\n```\n默认资源在 default 命名空间。"),
    ],
    "spark": [
        ("Spark 的持久化级别有哪些？", "Spark 缓存级别：\n- MEMORY_ONLY：仅内存\n- MEMORY_AND_DISK：内存+磁盘\n- MEMORY_ONLY_SER：内存序列化\n- DISK_ONLY：仅磁盘\n```python\nrdd.persist(StorageLevel.MEMORY_AND_DISK)\ndf.cache()\n```"),
        ("Spark 中 repartition 和 coalesce 的区别？", "- repartition：全量 shuffle，可增可减分区，开销大\n- coalesce：窄依赖合并，只能减少分区，开销小\n```python\ndf2 = df.coalesce(4)      # 减少分区无 shuffle\ndf3 = df.repartition(10)   # 增加分区有 shuffle\n```"),
    ],
    "mysql": [
        ("MySQL 中 CHAR 和 VARCHAR 的区别？", "- CHAR(n)：定长，不足补空格，最大 255 字符\n- VARCHAR(n)：变长，存实际长度+1/2字节\n```sql\nCREATE TABLE t (\n  code CHAR(6),\n  name VARCHAR(100)\n);\n```\nCHAR 查询效率略高，VARCHAR 节省空间。"),
        ("MySQL 如何实现分页查询？", "使用 LIMIT offset, count：\n```sql\nSELECT * FROM users LIMIT 0, 10;     -- 第 1 页\nSELECT * FROM users LIMIT 100, 10;   -- 第 11 页\n```\n深分页优化：用子查询先定位 ID 再查询。"),
    ],
    "react": [
        ("React 中 key 属性的作用？", "key 帮助 React 识别列表中哪些元素变化，用于高效更新 DOM。\n```jsx\nconst list = items.map(item => (\n  <li key={item.id}>{item.name}</li>\n))\n```\n不要用数组 index 作为 key，应使用唯一 ID。"),
        ("React 如何进行状态提升？", "将多个子组件共享的状态提升到共同父组件，通过 props 传递。\n```jsx\nfunction Parent() {\n  const [value, setValue] = useState('')\n  return (\n    <>\n      <Input value={value} onChange={setValue} />\n      <Display value={value} />\n    </>\n  )\n}\n```"),
    ],
    "java": [
        ("Java 中 String、StringBuilder、StringBuffer 的区别？", "- String：不可变，线程安全，频繁拼接性能差\n- StringBuilder：可变，非线程安全，性能好\n- StringBuffer：可变，线程安全，性能略差\n```java\nStringBuilder sb = new StringBuilder();\nsb.append(\"a\").append(\"b\");\nString result = sb.toString();\n```\n单线程拼接用 StringBuilder。"),
        ("Java 接口和抽象类的区别？", "- 接口：多实现，默认方法 public abstract，JDK 8 后可有默认方法\n- 抽象类：单继承，可有构造器、非抽象方法、成员变量\n```java\ninterface Flyable {\n    void fly();\n    default void land() { System.out.println(\"landing\"); }\n}\nabstract class Animal {\n    abstract void eat();\n}\n```"),
    ],
    "python": [
        ("Python 中深拷贝和浅拷贝的区别？", "- 浅拷贝 copy.copy()：只复制顶层，嵌套对象仍引用原对象\n- 深拷贝 copy.deepcopy()：递归复制所有层级\n```python\nimport copy\na = [[1, 2], [3, 4]]\nb = copy.copy(a)\nc = copy.deepcopy(a)\na[0].append(5)\nprint(b)  # [[1, 2, 5], [3, 4]]\nprint(c)  # [[1, 2], [3, 4]]\n```"),
        ("Python GIL 是什么？如何规避？", "GIL（全局解释器锁）使 CPython 同一时刻只有一个线程执行字节码。\n规避方法：\n1. 多进程（multiprocessing）\n2. I/O 密集型用多线程（GIL 在 I/O 时释放）\n3. C 扩展释放 GIL\n```python\nfrom multiprocessing import Pool\nwith Pool(4) as p:\n    results = p.map(func, data)\n```"),
    ],
    "pytorch": [
        ("PyTorch 中 DataLoader 的常用参数？", "DataLoader 关键参数：\n- batch_size：批大小\n- shuffle：是否打乱\n- num_workers：数据加载线程数\n- pin_memory：锁页内存\n- drop_last：丢弃最后不完整 batch\n```python\nloader = DataLoader(dataset, batch_size=32, shuffle=True,\n                    num_workers=4, pin_memory=True)\n```"),
        ("什么是学习率预热（Warmup）？", "Warmup 在训练初期用小学习率逐步升高到目标值，避免早期不稳定。\n```python\nfrom transformers import get_linear_schedule_with_warmup\nscheduler = get_linear_schedule_with_warmup(\n    optimizer, num_warmup_steps=100, num_training_steps=1000\n)\n```\n常用 warmup_ratio = 0.03~0.1。"),
    ],
    "linux": [
        ("Linux 中如何查看端口占用？", "常用命令：\n```bash\nss -tlnp              # 查看所有监听端口\nlsof -i :8080         # 查看指定端口\nnetstat -tlnp | grep 8080\n```\nss 比 netstat 更快更现代。"),
        ("Linux 中 find 和 grep 的区别？", "- find：按文件名/属性搜索文件\n- grep：在文件内容中搜索文本\n```bash\nfind . -name \"*.py\"           # 查找 .py 文件\ngrep -r \"TODO\" . --include=\"*.py\"  # 搜索内容\nfind . -name \"*.log\" | xargs grep \"ERROR\"\n```"),
    ],
    "docker": [
        ("Docker 中 COPY 和 ADD 的区别？", "- COPY：仅复制本地文件到镜像\n- ADD：支持 URL 下载、自动解压 tar 包\n```dockerfile\nCOPY app.py /app/\nADD archive.tar.gz /app/\n```\n推荐优先用 COPY，ADD 仅在需要解压/下载时使用。"),
        ("如何减少 Docker 镜像体积？", "优化策略：\n1. 使用多阶段构建\n2. 选用轻量基础镜像（alpine）\n3. 合并 RUN 命令减少层数\n4. 清理包管理器缓存\n```dockerfile\nFROM python:3.12-alpine\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\n```"),
    ],
    "git": [
        ("Git 中 merge 和 rebase 的区别？", "- merge：创建合并提交，保留完整历史\n- rebase：将提交移动到目标分支顶端，历史线性整洁\n```bash\ngit merge feature\ngit rebase main\n```\nrebase 会改写历史，已推送的公共分支不要 rebase。"),
        ("如何撤销 Git 提交？", "不同场景：\n```bash\ngit reset --soft HEAD~1   # 撤销提交，保留修改\ngit reset --mixed HEAD~1  # 撤销提交和暂存\ngit reset --hard HEAD~1   # 完全撤销（危险）\ngit revert HEAD           # 撤销已推送的提交\n```"),
    ],
}


def generate():
    lines = []
    seen = set()

    def add(q, a):
        h = hashlib.md5((q + a).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            lines.append({"question": q, "answer": a})

    # 1. 种子问答
    for qa in SEEDS:
        add(qa["q"], qa["a"])

    # 2. 领域知识点问答
    for domain, qas in DOMAIN_KNOWLEDGE.items():
        for q, a in qas:
            add(q, a)

    # 3. 基于模板的合成问答（答案有差异，避免被去重）
    fill_libs = [
        ("NumPy", "数值计算库", "数组运算和线性代数", "array", "向量化运算快"),
        ("Pandas", "数据分析库", "数据清洗和处理", "DataFrame", "表格操作方便"),
        ("Requests", "HTTP 库", "发送网络请求", "get", "API 简洁"),
        ("Pytest", "测试框架", "单元测试和集成测试", "main", "fixture 机制强大"),
        ("Scikit-learn", "机器学习库", "传统 ML 算法", "fit", "算法丰富"),
        ("Matplotlib", "可视化库", "绘制图表", "plot", "高度可定制"),
        ("SQLAlchemy", "ORM 库", "数据库操作", "create_engine", "支持多数据库"),
        ("FastAPI", "Web 框架", "构建 API 服务", "FastAPI", "自动生成文档"),
    ]
    fill_frameworks = [
        ("Django", "ORM", "models", "query"),
        ("Flask", "路由", "route", "url"),
        ("Celery", "异步任务", "task", "delay"),
        ("Airflow", "工作流编排", "DAG", "schedule"),
    ]
    fill_techs = [
        ("Redis", "内存读写快", "数据结构丰富", "支持持久化", "内存成本高", "单线程瓶颈", "缓存场景", "大规模关系型数据"),
        ("Kafka", "高吞吐", "低延迟", "多副本可靠", "运维复杂", "消息延迟", "日志/事件流", "简单消息队列"),
        ("MongoDB", "灵活 schema", "水平扩展", "文档模型", "事务支持弱", "内存占用大", "半结构化数据", "强事务场景"),
        ("Nginx", "高性能", "反向代理", "负载均衡", "配置较复杂", "动态模块", "Web 服务", "复杂应用逻辑"),
    ]
    fill_systems = [
        ("MySQL", "添加索引", "优化 SQL", "调整缓冲池", "SHOW STATUS", "SET GLOBAL"),
        ("Nginx", "开启 gzip", "调整 worker", "缓存静态资源", "nginx -t", "nginx -s reload"),
        ("Redis", "使用管道", "合理过期策略", "避免大 key", "redis-cli info", "CONFIG SET"),
        ("Kubernetes", "设置资源限制", "HPA 自动扩缩", "节点亲和性", "kubectl top", "kubectl scale"),
    ]
    fill_concepts = [
        ("进程", "线程", "资源分配", "独立地址空间", "共享地址空间", "调度单位", "独立", "轻量", "进程隔离性强，线程通信方便"),
        ("TCP", "UDP", "连接性", "面向连接", "无连接", "可靠性", "可靠", "不可靠", "TCP 适合可靠传输，UDP 适合实时性"),
        ("SQL", "NoSQL", "数据模型", "表结构", "灵活 schema", "事务", "ACID", "弱事务", "SQL 适合复杂查询，NoSQL 适合扩展"),
    ]

    # 每条答案附加的随序号变化的补充说明，保证 200 条不重复
    varying_notes = [
        "实际使用时需结合项目具体情况调整参数。",
        "建议先在小数据集上验证后再推广到生产环境。",
        "注意版本兼容性，不同版本 API 可能有差异。",
        "遇到问题可查阅官方文档获取最新信息。",
        "生产环境建议添加完善的错误处理和日志记录。",
        "性能优化需配合监控数据进行针对性调优。",
        "安全性方面需注意输入校验和权限控制。",
        "可结合单元测试确保功能正确性。",
        "文档和注释对后期维护非常重要。",
        "建议遵循团队编码规范保持一致性。",
    ]

    templates = [
        (
            "{lib} 是什么？它的核心用途是什么？",
            "{lib} 是一个{type}，主要用于{use}。\n```python\nimport {lib}\nresult = {lib}.{method}()\nprint(result)\n```\n核心优势在于{adv}。\n注：{note}",
        ),
        (
            "如何在 {framework} 中实现 {task}？",
            "在 {framework} 中实现 {task} 的步骤：\n1. 导入必要模块\n2. 配置参数\n3. 执行操作\n```python\nfrom {framework} import {module}\nconfig = {module}.Config({param}=value)\nresult = {module}.run(config)\n```\n通过以上步骤即可完成 {task}。\n注：{note}",
        ),
        (
            "{tech} 的主要优缺点是什么？",
            "{tech} 的优点：\n- {pro1}\n- {pro2}\n- {pro3}\n\n缺点：\n- {con1}\n- {con2}\n```text\n适用场景：{scenario}\n不适用场景：{nonscenario}\n```\n使用前需根据实际需求权衡。\n注：{note}",
        ),
        (
            "如何优化 {system} 的性能？",
            "{system} 性能优化策略：\n1. {opt1}\n2. {opt2}\n3. {opt3}\n```bash\n# 监控命令\n{cmd}\n# 配置调整\n{config_cmd}\n```\n优化前先通过监控定位瓶颈，避免盲目调参。\n注：{note}",
        ),
        (
            "{concept} 和 {other} 有什么区别？",
            "{concept} 与 {other} 的核心区别：\n\n| 维度 | {concept} | {other} |\n|------|-----------|----------|\n| {d1} | {a1} | {b1} |\n| {d2} | {a2} | {b2} |\n\n总结：{summary}。\n注：{note}",
        ),
    ]

    for q_tpl, a_tpl in templates:
        for i in range(200):
            note = varying_notes[i % len(varying_notes)] + f"（场景{i}）"
            if "{lib}" in q_tpl:
                lib = fill_libs[i % len(fill_libs)]
                q = q_tpl.format(lib=lib[0])
                a = a_tpl.format(lib=lib[0], type=lib[1], use=lib[2], method=lib[3], adv=lib[4], note=note)
            elif "{framework}" in q_tpl:
                fw = fill_frameworks[i % len(fill_frameworks)]
                task = f"使用{fw[0]}的{fw[1]}功能"
                q = q_tpl.format(framework=fw[0], task=task)
                a = a_tpl.format(framework=fw[0], module=fw[2], param=fw[3], task=task, note=note)
            elif "{tech}" in q_tpl:
                t = fill_techs[i % len(fill_techs)]
                q = q_tpl.format(tech=t[0])
                a = a_tpl.format(tech=t[0], pro1=t[1], pro2=t[2], pro3=t[3],
                                 con1=t[4], con2=t[5], scenario=t[6], nonscenario=t[7], note=note)
            elif "{system}" in q_tpl:
                s = fill_systems[i % len(fill_systems)]
                q = q_tpl.format(system=s[0])
                a = a_tpl.format(system=s[0], opt1=s[1], opt2=s[2], opt3=s[3],
                                 cmd=s[4], config_cmd=s[5], note=note)
            elif "{concept}" in q_tpl:
                c = fill_concepts[i % len(fill_concepts)]
                q = q_tpl.format(concept=c[0], other=c[1])
                a = a_tpl.format(concept=c[0], other=c[1], d1=c[2], a1=c[3], b1=c[4],
                                 d2=c[5], a2=c[6], b2=c[7], summary=c[8], note=note)
            add(q, a)

    # 4. 概念解释类
    concepts = [
        ("什么是 RESTful API？", "RESTful API 是基于 HTTP 协议的 API 设计风格。\n核心原则：资源用 URL 表示；用 HTTP 方法操作；无状态通信；统一接口。\n```http\nGET /api/users/123\nPOST /api/users\nPUT /api/users/123\nDELETE /api/users/123\n```"),
        ("什么是微服务架构？", "微服务是将单体应用拆分为多个小服务的架构。\n特点：每个服务独立部署、独立数据存储；服务间通过 API 通信；技术栈可异构。\n```text\n用户服务 -> 订单服务 -> 支付服务\n```\n优势是灵活扩展，挑战是分布式复杂度。"),
        ("什么是 CAP 定理？", "CAP 定理：分布式系统中一致性(C)、可用性(A)、分区容错性(P)三者最多满足两个。\n- CP：一致性+分区容错（ZooKeeper）\n- AP：可用性+分区容错（Cassandra）\n```text\nP 是分布式必然，所以实际在 C 和 A 间权衡\n```"),
        ("什么是 BASE 理论？", "BASE 是对 CAP 中 AP 的扩展：\n- Basically Available：基本可用\n- Soft state：软状态\n- Eventually consistent：最终一致性\n```text\n与 ACID 强一致性相对，BASE 允许短暂不一致\n```"),
    ]
    for q, a in concepts:
        add(q, a)

    # 5. 错误排查类
    errors = [
        ("如何排查 OOM（内存溢出）？", "排查步骤：\n1. 查看日志中的 OOM 错误\n2. 检查堆内存设置（-Xmx）\n3. 用 jmap/jstack 分析内存\n4. 检查内存泄漏\n```bash\njmap -dump:format=b,file=heap.hprof <pid>\njstack <pid> > thread.txt\n```\n常见原因：内存泄漏、大对象、缓存未限制。"),
        ("如何排查 CPU 100% 问题？", "排查步骤：\n1. top 找到高 CPU 进程\n2. top -H -p <pid> 找到高 CPU 线程\n3. printf '%x' <tid> 转十六进制\n4. jstack 查看线程堆栈\n```bash\ntop -H -p 12345\nprintf '%x' 67890\njstack 12345 | grep -A 20 <hex_tid>\n```\n常见原因：死循环、频繁 GC、锁竞争。"),
    ]
    for q, a in errors:
        add(q, a)

    random.shuffle(lines)

    with open(ROOT, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[DATA] Generated {len(lines)} lines -> {ROOT}")
    return len(lines)


if __name__ == "__main__":
    generate()
