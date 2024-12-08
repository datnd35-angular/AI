

## Những mốc quan trọng trong lịch sử AI
- 1950 - Alan Turing và bài kiểm tra Turing: Turing đặt câu hỏi "Máy móc có thể suy nghĩ không?" và giới thiệu tiêu chuẩn (Turing Test) để đánh giá trí tuệ máy móc.
- 1956 - Hội nghị Dartmouth: Thuật ngữ "AI" được ra đời, đánh dấu AI trở thành lĩnh vực nghiên cứu chính thức.
- 1960s-70s - AI Winter: Sự phát triển AI đình trệ do thiếu dữ liệu, công nghệ và tài trợ.
- 1997 - Deep Blue đánh bại Garry Kasparov: Đánh dấu sự trở lại của AI với sức mạnh tính toán gia tăng.
- 2006 - Deep Learning: Geoffrey Hinton hồi sinh nghiên cứu mạng nơ-ron với kỹ thuật deep learning, nhờ dữ liệu lớn và khả năng xử lý cao.
- 2011 - IBM Watson thắng Jeopardy!: Tiến bộ trong xử lý ngôn ngữ tự nhiên.
- 2012 - Nghiên cứu của Google và Stanford: Mạng nơ-ron sâu xuất sắc trong nhận diện hình ảnh (ví dụ: mèo).
- 2017 - Transformers của Google Brain: Đột phá xử lý ngôn ngữ tự nhiên nhờ cơ chế tự chú ý (self-attention).
- 2018 - OpenAI GPT: Công nghệ AI sinh tạo (Generative AI) sử dụng Transformers, dẫn đến ChatGPT năm 2022.

## Demystifying AI, Data science, Machine learning, and Deep learning

- ***AI (Trí tuệ nhân tạo):*** Lĩnh vực rộng, tập trung làm cho máy móc thông minh, cho phép chúng học hỏi và phát triển kỹ năng mới.
- ***Machine Learning (Học máy):*** Một nhánh quan trọng của AI, sử dụng dữ liệu để dự đoán kết quả bằng cách xác định các mối quan hệ phức tạp vượt ra ngoài sự tương quan đơn thuần. Ví dụ: dự đoán phim yêu thích dựa trên lịch sử xem phim, hoặc tính điểm tín dụng từ giao dịch tài chính.
- ***Data Science (Khoa học dữ liệu):*** Bao gồm AI và Machine Learning nhưng cũng sử dụng các phương pháp thống kê truyền thống như trực quan hóa dữ liệu và suy luận thống kê. Mục tiêu là trích xuất giá trị kinh doanh từ dữ liệu, ví dụ: phát triển thuật toán dự đoán đơn hàng hoặc phân tích dữ liệu khách hàng để tìm hiểu hành vi.

## Weak vs Strong AI
- ***Narrow AI (AI hẹp):*** AI được thiết kế để thực hiện một nhiệm vụ cụ thể, như dự đoán phim dựa trên lịch sử xem. Loại AI này đã được tích hợp vào cuộc sống hàng ngày và mang lại giá trị lớn cho doanh nghiệp.

- ***Semi-strong AI (AI bán mạnh):*** Các mô hình như ChatGPT (ra mắt năm 2022) đã đạt đến cấp độ này, có khả năng thực hiện nhiều nhiệm vụ khác nhau, từ viết nội dung, tạo công thức Excel, đến giải toán. Chúng có thể vượt qua bài kiểm tra Turing, tạo phản hồi tương tự như con người.

- ***Artificial General Intelligence (AGI):*** Được gọi là AI mạnh, đây là cấp độ AI có khả năng thông minh toàn diện, vượt trội hơn con người trong nhiều nhiệm vụ. AGI có thể tự tạo ra kiến thức khoa học mới từ dữ liệu sẵn có. CEO OpenAI, Sam Altman, dự đoán AGI có thể xuất hiện trong tương lai gần.

# Yếu tố chính để tạo ra mô hình AI

## 1. Dữ liệu cấu trúc và dữ liệu không có cấu trúc
***Dữ liệu có hai loại chính: dữ liệu có cấu trúc (organized thành hàng và cột như bảng Excel) và dữ liệu phi cấu trúc (như tệp văn bản, hình ảnh, video, âm thanh, không tổ chức thành hàng cột).***

- 80-90% dữ liệu trên thế giới là phi cấu trúc, khó phân tích hơn dữ liệu có cấu trúc.
- Trước đây, dữ liệu có cấu trúc được đánh giá cao hơn vì dễ phân tích.
- Tuy nhiên, nhờ AI hiện đại, dữ liệu phi cấu trúc giờ đây có thể được phân tích để mang lại giá trị lớn, giúp các công ty như Meta và Google tạo ra những cơ hội mới.

## 2. Cách thu thập dữ liệu
- ```MNIST (Modified National Institute of Standards and Technology database)``` nó là ```database``` ví dụ cơ bản cho ```machine learning```, chứa ```70.000 ảnh xám kích thước 28x28 pixel``` về chữ số viết tay.
 Mục tiêu là huấn luyện AI nhận diện chữ số dù có sự khác biệt về nét chữ và được xem như là ```"Hello World"``` trong ```machine learning``` giúp người mới bắt đầu làm quen.
- Mỗi pixel có giá trị từ 0 (trắng) đến 255 (đen) và được lưu trữ dưới dạng nhị phân (0 và 1).
- Máy tính sử dụng dữ liệu pixel này để tìm mẫu, học cách phân biệt các số từ 0 đến 9.
- Ví dụ ```MNIST``` minh họa cách biến thông tin thực tế thành dữ liệu số để máy tính xử lý.
- Tương tự:
  - Video là tập hợp các ảnh.
  - Âm thanh và văn bản cũng được biểu diễn dưới dạng nhị phân.

***Điểm ảnh (Pixel) là gì?***
- Pixel (Picture Element): Là đơn vị nhỏ nhất của một màn hình. Mỗi pixel có thể hiển thị một màu cụ thể tại một thời điểm.
- Hình ảnh trên màn hình được tạo thành từ hàng triệu pixel, mỗi pixel kết hợp màu sắc để tạo nên hình ảnh tổng thể.

## 3. Dữ liệu có nhãn và dữ liệu không nhãn
***Dữ liệu có nhãn (Labeled Data):***
- Mỗi mục trong bộ dữ liệu được phân loại rõ ràng (ví dụ: ảnh được gắn nhãn là "chó" hoặc "không phải chó").
- Quá trình gắn nhãn thủ công rất tốn thời gian và chi phí, nhưng đảm bảo mô hình AI được huấn luyện chính xác hơn, giúp tăng hiệu suất trong ứng dụng thực tế.

***Dữ liệu không nhãn (Unlabeled Data):***
- Dữ liệu không được phân loại trước (ví dụ: 10,000 ảnh mà không qua gắn nhãn).
- AI sẽ tự học từ dữ liệu này mà không cần gắn nhãn ban đầu.
- Đây là cách hiệu quả với dữ liệu lớn, nhưng đòi hỏi kỹ thuật phức tạp để giúp AI tự tìm ra cấu trúc hoặc mẫu trong dữ liệu.

## Metadata: Dữ liệu miêu tả dữ liệu 
***Lý do***
- Lượng dữ liệu lớn hơn và chất lượng cao hơn (ví dụ: ảnh từ điện thoại ngày nay sắc nét hơn nhiều so với trước đây) đã thúc đẩy AI phát triển mạnh mẽ.
- Phần lớn dữ liệu hiện nay là không có cấu trúc và quá lớn để gắn nhãn thủ công.

***Metadata***
- Metadata (dữ liệu mô tả dữ liệu) rất quan trọng để quản lý và làm rõ thông tin trong tập dữ liệu lớn.
- Metadata bao gồm các thông tin như loại tệp, ngày tạo, tác giả, kích thước tệp, và các chi tiết khác.


# Key AI technique

***1. Machine Learning***

Chúng ta đã khám phá ```các khái niệm cơ bản của AI```, lịch sử của nó và sự phân biệt giữa ```AI yếu và AI mạnh```. Chất lượng và lượng dữ liệu là yếu tố quan trọng trong việc xây dựng AI.

Tiếp theo, chúng ta sẽ khám phá ```machine learning (ML)```, một nhánh quan trọng của AI gần đây đã đạt được thành công lớn. Ý tưởng của ML là thiết kế một hệ thống có khả năng học hỏi và cải thiện thông qua quá trình thử và sai.

ML có thể được coi như một sinh viên được giáo viên hướng dẫn. Giáo viên là người cung cấp dữ liệu đào tạo và hướng dẫn sinh viên cách giải quyết vấn đề. Mục tiêu là giúp sinh viên học cách giải quyết các vấn đề mới với dữ liệu chưa từng thấy trước đó.

Một ví dụ thực tế về ứng dụng ML là trong lĩnh vực bất động sản, nơi một đại lý bất động sản phát triển một ứng dụng di động để cung cấp ước tính giá bán nhà. Các yếu tố như diện tích, số phòng, khoảng cách từ trung tâm, v.v. được sử dụng để dự đoán giá nhà. Dữ liệu này giúp tạo đầu vào cho mô hình ML, dự đoán giá nhà Y dựa trên các dữ liệu quá khứ và đặc điểm của nhà.

Mô hình ML thành công khi nó được đào tạo tốt với dữ liệu chất lượng và đủ lớn, giúp cải thiện độ chính xác trong dự đoán và phân tích.

Hi vọng bài học này đã giúp bạn hiểu rõ hơn về ML và cách nó hoạt động. Trong bài học tiếp theo, chúng ta sẽ thảo luận về các loại mô hình ```machine learning``` khác nhau.

***2. Supervised, Unsupervised, and Reinforcement***

- ***Supervised learning*** nổi bật khi làm việc với dữ liệu đã được gán nhãn. Ví dụ trong trường hợp trước, tập dữ liệu huấn luyện chỉ ra rằng một bức ảnh có chứa con chó. Đây là cách thuật toán có thể học để phân loại những bức ảnh mới là có hoặc không có con chó. Chúng tôi đã cung cấp cho mô hình ML một tập dữ liệu huấn luyện rộng lớn với dữ liệu đã gán nhãn, và nó biết phải tìm gì. Sau đó, khi chúng ta cung cấp một bức ảnh mới, nó có thể xác định liệu có con chó trong đó dựa trên phản hồi trong quá trình huấn luyện. Đây là một vấn đề phân loại được giải quyết bởi supervised ML.

- Ứng dụng chính khác của supervised learning là dự đoán kết quả. Nhớ khi chúng ta thảo luận về ứng dụng di động của môi giới bất động sản trong bài học trước không? Dữ liệu tập hợp các giao dịch mua bán nhà trong quá khứ có cả giá cả và đặc điểm của các ngôi nhà. Đây cũng là một ví dụ của supervised learning, nhưng nó được sử dụng cho dự đoán, còn gọi là vấn đề hồi quy. Khi một người dùng nhập vào các đặc điểm của một ngôi nhà mới vào ứng dụng, ứng dụng sẽ dự đoán giá của ngôi nhà mới dựa trên các đặc điểm và thông tin lịch sử của các ngôi nhà tương tự.

- ***Unsupervised learning*** chúng ta xử lý dữ liệu mà không có nhãn. Ví dụ, trong tập hợp 10.000 bức ảnh động vật, một nửa là chó và nửa còn lại là các loài động vật khác, nhưng chúng ta không cung cấp cho mô hình machine learning bất kỳ nhãn nào hay chỉ dẫn nào. Nó sẽ cần phải quét qua tất cả các bức ảnh và tìm kiếm các mẫu. Cuối cùng, nó sẽ có thể phân biệt các nhóm cụ thể và phân loại chúng. Mô hình ML không chỉ ra nội dung của những hình ảnh này mà chỉ đơn giản chỉ ra rằng chúng có đặc điểm tương tự.

- ***Reinforcement learning*** hoạt động mà không có dữ liệu gán nhãn và được áp dụng trong các tình huống cụ thể, nơi máy tính tìm ra cách đạt được mục tiêu đã định. Reinforcement learning chia sẻ với supervised ML trong việc hiểu mục tiêu mong muốn, nhưng khác biệt là chúng tôi không cung cấp dữ liệu gán nhãn cho mô hình ML. Thay vào đó, chúng tôi tạo ra các quy tắc. Mô hình ML học thông qua thử nghiệm và sai lầm trong các thông số cụ thể của các quy tắc được tạo ra. Reinforcement learning phát triển mạnh mẽ trong robot học và hệ thống đề xuất trực tuyến. Ví dụ, các nền tảng như Netflix sử dụng reinforcement learning để cải thiện hệ thống đề xuất. Thay vì sử dụng các phương pháp truyền thống dựa trên dữ liệu đã gán nhãn, mà chỉ ra các chương trình truyền hình mà người dùng có thể thích, reinforcement learning sử dụng hệ thống mà mô hình học hỏi từ phản hồi của người dùng thông qua tương tác. Về lâu dài, mô hình tinh chỉnh các dự đoán để phù hợp hơn với sở thích cá nhân của từng người dùng, liên tục thích ứng với các mẫu xem mới và hành vi.

***3. Deep learning***


- ```Deep learning``` là một phân nhánh hấp dẫn của ```machine learning``` được lấy cảm hứng từ cách bộ não con người hoạt động.

- Hình ảnh AI tạo ra là gì? Ban đầu, chúng ta thấy một ngày nắng và bãi biển đông đúc, đúng không? Khi nhìn kỹ hơn, chúng ta thấy trẻ em đang chơi cát xung quanh lâu đài cát khổng lồ ở trung tâm.

- Não bộ của chúng ta xử lý thông tin ở nhiều giai đoạn và ở các độ sâu khác nhau. Ban đầu, việc nhìn vào một bức ảnh mang lại ấn tượng thô sơ rộng rãi về bối cảnh của cảnh đó. Khi chúng ta dành thêm thời gian và sự chú ý vào chi tiết, chúng ta có thể xử lý và quan sát nhiều thông tin hơn, và sâu hơn trong ngữ cảnh của deep learning.

- Một mạng nơ-ron hoạt động tương tự. Cơ chế của nó có thể được tưởng tượng như sau: lớp đầu vào của mạng nơ-ron nhân tạo nhận dữ liệu thô (như hình ảnh về một ngày nắng trên bãi biển). Khi dữ liệu đi qua nhiều lớp của mạng, các lớp trung gian bắt đầu nhận dạng các đặc điểm phức tạp như hình dạng hoặc các đối tượng cụ thể. Càng sâu, mạng càng tổng hợp các đặc điểm ở mức cao hơn, thể hiện các khía cạnh phức tạp của dữ liệu đầu vào.

- Mục tiêu là nhận dạng các khuôn mặt kỳ quặc do AI tạo ra hoặc hiểu các hình ảnh phức tạp khác. Quá trình này mất thời gian để xử lý thông tin và đưa ra kết luận.

***Neural network*** hay mạng nơ-ron là sự mô phỏng của mạng nơ-ron sinh học, nhưng hoạt động khác nhiều. Các lớp ẩn hoặc lớp trung gian xử lý thông tin đầu vào, có thể có một hoặc nhiều lớp ẩn. Tăng số lượng lớp làm tăng tính phức tạp của mạng và khả năng học hỏi.

- Trong quá trình huấn luyện, mạng nơ-ron tiếp nhận hình ảnh của một chữ số viết tay. Mỗi pixel của hình ảnh được coi như một nút đầu vào. Ví dụ mNIST có kích thước hình ảnh là 28x28 pixel, vì vậy lớp đầu vào có 784 nút đầu vào. Các nút này chứa một số dựa trên độ sáng hoặc tối của pixel.

- Mỗi lớp của mạng nơ-ron nhân tạo bao gồm các nơ-ron (hoặc nút) có nhiệm vụ xử lý và biến đổi thông tin nhận được. Các lớp của mạng nơ-ron tạo ra một hệ thống mạnh mẽ của nhận dạng mẫu và diễn giải dữ liệu, tương tự một số khía cạnh của quá trình nhận thức của con người.

***Deep learning*** là một công cụ cách mạng trong AI nhờ khả năng phân tích các tập dữ liệu lớn, nhiều chiều và nhận dạng các mẫu phức tạp với độ chính xác cao, đó là điều làm cho sự tiến bộ đáng kinh ngạc của AI ngày nay trở nên khả thi.

# Key AI technique

***1. Robotics***

Câu chuyện về các sinh vật cơ khí có từ thời cổ đại. Ví dụ, thần thoại về Talos mô tả một người khổng lồ bằng đồng được Hephaestus, vị thần Hy Lạp của sáng tạo và thợ rèn, tạo ra với nhiệm vụ bảo vệ Crete. Hình đồng này tuần tra đảo ba lần mỗi ngày, ném đá vào các tàu thuyền của kẻ thù đến gần bờ.

Trong thời Trung cổ phương Tây và Hồi giáo, các thiết bị như các tự động của Al-Jazari, bao gồm đồng hồ nước và máy móc lập trình, thể hiện tài năng kỹ thuật sớm. Leonardo da Vinci, một trong những người tiên phong lớn của thời kỳ Phục hưng Italy, đã phác thảo thiết kế của một hiệp sĩ cơ khí có thể di chuyển giới hạn, tiên đoán về các rô-bốt hình người hiện nay.

Ý tưởng về rô-bốt và lĩnh vực rô-bốt không mới. Con người đã bị cuốn hút bởi chúng từ lâu. Đó là một hiện tượng thực sự trong văn hóa đại chúng, thu hút trí tưởng tượng của nhiều thế hệ. Ngày nay, nhờ những tiến bộ công nghệ nhanh chóng và chủ yếu là sự phát triển của AI, việc tạo ra các máy thông minh mà trước đây chúng ta chỉ có thể tưởng tượng và mô tả trong phim là khả thi.

Rô-bốt là nhánh của công nghệ liên quan đến việc thiết kế, chế tạo, vận hành và sử dụng rô-bốt – những máy móc có thể thực hiện các nhiệm vụ tự động hoặc với khả năng giống con người. Các kỹ sư cơ khí chịu trách nhiệm thiết kế và xây dựng cấu trúc vật lý của rô-bốt, bao gồm cơ chế di chuyển của nó. Các kỹ sư điện tử và điện học thiết kế các hệ thống cho phép rô-bốt hoạt động và kiểm soát hành động của nó. Trong khi các điện tử cung cấp năng lượng cho hệ thống, nhiều loại AI điều khiển quyết định và hành vi của rô-bốt. Rô-bốt được trang bị nhiều loại cảm biến và camera để thu thập dữ liệu cần thiết và nhận thức môi trường xung quanh.

Rô-bốt được thiết kế để mô phỏng và bổ sung khả năng của con người, do đó yêu cầu một cách tiếp cận liên ngành kết hợp nhiều công nghệ AI khác nhau. Ví dụ, một rô-bốt Automatas có thể sử dụng thị giác máy tính để phát hiện đối tượng và hiểu môi trường, định vị đồng thời và lập bản đồ cho việc định vị và lập bản đồ, học tăng cường để ra quyết định, và một mô hình xử lý ngôn ngữ tự nhiên để hiểu và tạo ngôn ngữ của con người.

Một số ứng dụng thú vị của rô-bốt bao gồm rô-bốt y tế có thể thực hiện can thiệp y tế chính xác và phẫu thuật phức tạp, rô-bốt tự lái, rô-bốt thu hoạch, rô-bốt làm sạch, rô-bốt không gian, rô-bốt tìm kiếm và cứu hộ, và rô-bốt an ninh và giám sát. Tương lai đã đến, và AI đứng đầu trong tất cả các đổi mới này.

***2. Computer vision***

- IBM định nghĩa tầm nhìn máy tính là một lĩnh vực AI sử dụng học máy và mạng nơ-ron để dạy máy tính hiểu thông tin có ý nghĩa từ hình ảnh và video kỹ thuật số. Nếu AI đóng vai trò là não bộ, thì tầm nhìn máy tính là "mắt" của máy tính. Con người dễ dàng hiểu môi trường xung quanh với sự tinh tế của nó, phân biệt các vật thể chuyển động, hình dạng thay đổi và màu sắc khác nhau. Nghiên cứu tầm nhìn máy tính là nỗ lực lớn để phát triển các mô hình AI phức tạp cho phép máy tính hiểu thông tin thế giới thực.

- Máy tính tiêu thụ dữ liệu qua hình ảnh và video. Hình ảnh đơn giản hơn vì chúng ghi lại một khoảnh khắc tĩnh, trong khi video phức tạp hơn với các dãy hình ảnh liên tục (ví dụ, 30 khung hình mỗi giây) yêu cầu xử lý. Có bốn nhóm chính của các mô hình tầm nhìn máy tính:

- ***Mạng nơ-ron tích chập (CNNs):*** CNNs là nền tảng của tầm nhìn máy tính vì chúng rất hiệu quả với dữ liệu có chiều cao. Chúng tổ chức các yếu tố trong hình ảnh dựa trên tầm quan trọng và độ sâu. CNNs học các đặc trưng cơ bản như cạnh của vật thể ở các lớp đầu tiên và dần dần nắm bắt các đặc trưng phức tạp như hình dạng và các đối tượng ở các lớp sâu hơn.

- ***Transformers:*** Đôi khi AI sử dụng kiến trúc transformer cho mục đích tầm nhìn máy tính.

- ***Mạng đối kháng sinh tạo (GANs):*** Chủ yếu tạo ra hình ảnh sống động.

- ***Mạng chuyên biệt như U-Net:*** Xuất sắc trong việc phân đoạn hình ảnh y tế.

- ```Computer vision``` có rất nhiều ứng dụng thực tế như xe tự lái, hình ảnh y tế, an ninh, giám sát và các lĩnh vực trước đây đã được thảo luận. Tầm nhìn máy tính không nhất thiết phải là một phần của rô-bốt để trở thành một sản phẩm cực kỳ hữu ích. Ví dụ, phần mềm nhận dạng khuôn mặt chỉ là một mô hình được tích hợp trong sản phẩm phần mềm mà không cần thân rô-bốt. Các tiến bộ quan trọng nhất trong tầm nhìn máy tính hiện nay đang diễn ra trong thực tế ảo, cách mạng hóa các ngành công nghiệp như giáo dục, giải trí và giao tiếp từ xa. Trong bài học tiếp theo, chúng ta sẽ thảo luận về các phương pháp học máy truyền thống.

***3. Traditional ML***

Sản phẩm như ChatGPT, xe tự lái của Tesla và rô-bốt gây sự chú ý, nhưng phần lớn giá trị mà AI tạo ra hiện nay lại đến từ các ứng dụng kinh doanh ít nổi bật hơn. Các tổ chức tài chính sử dụng máy học để phát hiện hoạt động gian lận hoặc dự đoán khả năng khách hàng trả nợ thế chấp. Các công ty bảo hiểm sử dụng máy học để định giá chính xác hơn các gói bảo hiểm. Các công ty bán lẻ dự đoán nhu cầu và tối ưu hóa đơn đặt hàng. Các gã khổng lồ thương mại điện tử như Amazon tối ưu hóa giá và tỷ lệ chuyển đổi thông qua các thuật toán máy học. Những ứng dụng này và nhiều trường hợp khác đã tồn tại hơn một thập kỷ. Rất ít doanh nghiệp thất bại khi sử dụng AI. Chúng ta cần nhớ rằng mặc dù AI tổng quát đang trở nên phổ biến và hứa hẹn, nhưng các ứng dụng AI truyền thống tôi đã đề cập tiếp tục ảnh hưởng đến doanh nghiệp và mang lại giá trị đáng kể.

***4. Generative AI***

```ChatGPT``` trở thành hiện tượng toàn cầu khi ra mắt, gây sự phấn khích rộng rãi về tiềm năng vượt trội của AI. Nhánh AI đằng sau những mô hình ấn tượng như ```ChatGPT``` được gọi là ```generative AI```. ```Generative AI``` có khả năng tạo ra dữ liệu hoặc nội dung mới từ các mẫu học được từ dữ liệu đào tạo. Các cuộc trò chuyện với ChatGPT là mở và không giới hạn, với mô hình này tạo ra phản hồi mới và độc đáo theo thời gian thực. ```Dall-E``` của OpenAI, mô hình tạo ảnh, cũng sử dụng khả năng này để tạo ra hình ảnh mới từ các mô tả chi tiết. Các kỹ thuật generative AI khác nhau như ```large language models, diffusion models, generative adversarial networks, neural radiance fields,``` và các mô hình lai cũng đang được phát triển. Generative AI có thể tạo ra nội dung dưới nhiều định dạng khác nhau như văn bản, hình ảnh, video, âm thanh, dữ liệu, mã, thiết kế, và 3D. Với sự phát triển của công nghệ, generative AI sẽ có ảnh hưởng lớn đến thế giới doanh nghiệp, và việc hiểu rõ về nó sẽ giúp chúng ta đi đầu trong sự đổi mới này.

