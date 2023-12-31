import os

# requirements.txt dosyasındaki kütüphaneleri yükle
os.system('pip install -r requirements.txt')
import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import plot_confusion_matrix

random.seed(42)

st.image('logos.png', width=350, use_column_width=True)

st.markdown("<h1 style='text-align: center;'>Yıldız Teknik Üniversitesi</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Kimya ve Metalurji Fakültesi</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Matematik Mühendisliği Bölümü</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Bitirme Çalışması</h2>", unsafe_allow_html=True)

st.write("")
st.write("")
st.markdown("<h2 style='text-align: center;'>Makine Öğrenmesi Metotlarının Karşılaştırılması ve Hyperarametre Optimizasyonlarının Yapılması</h2>", unsafe_allow_html=True)

st.caption("Danışman: Dr. Öğretim Üyesi Fatih AYLIKÇI")
st.caption("Yalçın Furkan ÇELİK 18052010")


st.markdown("##### Önsöz")
st.write("""Bu proje 21. yüzyılın en önemli gelişmelerinden biri olan yapay zekanın temel taşlarından olan makine öğrenmesi algoritmalarının uygulamarını içerecektir. Kullanılan algoritmaların aynı veri seti üzerinde aynı hedef değişken için ne kadar farklı sonuçlar verdiği gösterilmiştir. Gelecek nesiller bu teknolojiyi çok daha iyi kullanacaktır. Projenin gerçek amacı da makine öğrenmesi algoritmalarının uygulanabilirliği göstermektir. Bu kapsamda projenin detaylarında adımlar kodları ile birlikte verilmiştir.  """)

st.markdown("##### Özet")
st.write(""" Makine öğrenmesi, bilgisayar sistemlerinin veri kullanarak öğrenme yeteneğine sahip olmasını sağlamaya odaklanan yapay zekanın bir alt dalıdır. Bu teknoloji, bilgisayar sistemlerinin belirlenmiş bir probleme yönelik çözüm sunabilmek için veri üzerinde öğrenme metotlarını uygulamaya çalışır. Her algoritmanın kendine özgü çözümü olduğu için aynı adımlar ve farklı algoritmalar kullanılarak aynı veya tamamen farklı sonuçların alınması mümkündür. Bu sebeple birden fazla deneme sürecinden ve algoritmaların optimğizasyonundan sonra en iyi makine öğrenmesi modelinin bulunması amaçlanır. 

Bu bitirme projesinde makine öğrenmesi algoritmalarının aynı veriseti üzerinde çalıştırılarak doğruluk sonuçlarının karşılaştırılması amaçlanır. Buna ek olarak karşılaştırmanın dinamik olarak test edilebilmesi için de uygulamalı olarak gösterilebilecek bir web sitesinde uygulanarak projenin tamamlanması hedeflenmektedir.

Projenin sonunda veriye uygun modelleme konusunda gelişim sağlanarak optimum modelin elde edilmesine katkı sağlanması beklenmektedir. Bu aşama makine öğrenmesinde doğruluğu arttırmak için yapılabilecek en önemli aşamalardan biri olduğu için büyük bir önem arz etmektedir.
""")

st.markdown("##### 1.Özgün Değer")
st.markdown("##### 1.1. Konunun Önemi, Projenin Özgün Değeri, Gerçekçi Kısıtlar ve Koşulların Belirlenmesi ")

st.markdown(""" 
**Makine Öğrenmesinin Temelleri:** Makine öğrenmesinin temellerinde veri, algoritma ve sonuçların analizi vardır. Bu bir döngü halindedir ve sonuçların iyileştirebilmek için veri manipülasyonundan başlanarak verinin iyileştirilmesi veriye öznitelik kazandırılması amaçlanır. Daha sonrasında algoritmanın geliştirilmesi vardır, önce doğru metot seçilerek başlanır ardından parametre optimizasyonu ile model geliştirilmeye çalışır. Bütün bunların sonucunda modelin başarısının en iyi seviyede olması gerekir. Buradaki kritik noktalardan birisi de modelin veriyi ezberlemiş olma ihtimalidir. Bu da eldeki veri için mükemmel sonuçlar çıkartırken mevcut veri dışında bir çalışma yapıldığında tamamı ile yanlış bir sonuç çıkartabilmesidir. Çalışmalar yapılırken bunun da göz ardı edilmemesi gerekir. \n
**Makine Öğrenmesinin Kullanım Alanları:** \n
•	Görüntü Tanıma, objeler üzerindeki desenleri tanıyarak görüntü tanıma çalışmalarında kullanılabilir.\n
•	Doğal Dil İşleme, metinler üzerinden özellikle duygu-durum analizi gibi çalışmalarda kullanılabilmektedir.\n
•	Finans, özellikle hisse senetlerinin fiyat tahminlemelerinde kullanımı oldukça yaygındır.\n
•	Tıbbi Teşhis, sağlık alanında ön tanılama sistemlerinde kullanılmaktadır.\n
**Çalışmanın Katkı Sağlayacağı Konular:** Makine öğrenmesi sürecinin aşamalarında müdahale edilerek başarıyı değiştiren en önemli noktanın model seçimi olduğunu göz ardı edemeyiz. Bu sebeple çalışmanın sonucunda sektöre ve literatüre doğru modelin seçilmesinde ardından da modelin geliştirilmesi noktasında katkı sağlanması beklenmektedir. \n
**Problem:** Bu çalışmanın temel aldığı problem makine öğrenmesinin uygulanabileceği alanlarda model seçimi ve modelin geliştirilmesi sürecinin yeteri kadar açıklanmamış olmasıdır.

""")

st.markdown("##### 1.2. Amaç ve Hedefler")
st.markdown("""
Bu projedeki amaç makine öğrenmesi sürecinin kolaylaştırılması ve doğruluğunun arttırılmaya çalışmaktadır. Bunun için makine öğrenmesi uygulamalarında kritik nokta olan model seçimi ve optimizasyonun iyileştirilmesine odaklanılmaktadır. Burada yapılmak istenen bu sürecin daha açık ve anlaşılır olarak gösterilerek bu teknolojinin gelişime katkı sağlamaktır.  \n
**Projenin amaçları şu şekildedir:**\n
•	Makine öğrenmesi çalışmasına uygun veri seti elde edilmesi.\n
•	Veri seti üzerinde python programlama ile veri manipülasyonu uygularak verinin modelleme için hazır hale getirilmesi. \n
•	Veriye ve projenin kapsamına uygun makine öğrenmesi algoritmalarının belirlenmesi.\n
•	Algoritmaların kendi içerisindeki çalışma şekillerinin tariflenmesi.\n
•	Algoritmaların veriye uygulanarak modelleme işlemine başlanması.\n
•	Modelin sonuçlarının analiz edilerek gerekli geliştirmelerin yapılması ve raporlanması.\n
•	Tüm bu sürecin uygulamalı olarak gösterilebileceği web arayüzünün uygulanması.\n
""")

st.markdown("#### 2. Yöntem")



with st.expander("Lojistik Regresyon Algoritması"):

    st.header("Logistik Regresyon ile Sınıflandırma")
    st.write("""
Lojistik Regresyon, bir sınıflandırma algoritmasıdır ve genellikle iki sınıflı (binary) problemlerde kullanılır. Temel amacı, bir veri noktasının iki sınıftan birine ait olup olmadığını tahmin etmektir. Sigmoid fonksiyonu kullanarak, giriş özelliklerinin ağırlıklı toplamını bir olasılık değerine dönüştürür ve bu olasılığı belirli bir eşik değeri üzerinden sınıflara atar.
             """)
    lojistik_regresyon_formul = r'''
    P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n)}}
    '''
    st.latex(lojistik_regresyon_formul)
    st.markdown("""
<div style="font-size: 18px; line-height: 1.6;">
    <p>Burada:</p>
    <ul>
        <li><b>P(Y=1):</b> Bir olayın gerçekleşme olasılığı,</li>
        <li><b>β₀, β₁, β₂, ..., βₙ:</b> Katsayılar,</li>
        <li><b>X₁, X₂, ..., Xₙ:</b> Giriş özellikleri.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.subheader("Kullanılacak Kütüphaneler")
    st.code("""
import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

            """)

    st.subheader("Verinin Okunması")
    st.code("""
df = pd.read_csv('titanic.csv')
df.head()
    """)
    import pandas as pd

    df = pd.read_csv('titanic.csv')
    st.write("Verinin ilk 5 satırı")
    st.dataframe(df.head())

    st.subheader("Verinin Modellenmeye Hazır Hale Getirilmesi")
    st.write("Kullanılmayacak kolonlar çıkartılıp cinsiyet ve Pclass bilgilerinin lojik hale getirilmesi")
    st.code("""
df = df[['Survived', 'Age', 'Sex', 'Pclass']]
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
df.dropna(inplace=True)
            """)
    df = df[['Survived', 'Age', 'Sex', 'Pclass']]
    df = pd.get_dummies(df, columns=['Sex', 'Pclass'])
    df.dropna(inplace=True)
    st.dataframe(df.head())
    st.write("Veriden Survived kolonu ayrıştırılıyor ve veri eğitim ve test verisi olarak ayrılıyor.")
    st.code("""
x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    """)
    x = df.drop('Survived', axis=1)
    y = df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)


    st.subheader("Modelin Eğitilmesi")
    st.write("Modelin eğitilmesinde lojistik regresyon metodu kullanılacaktır. Lojistik Regresyon sadece tahminleme yapmaz bunun yanı sıra olasılıkları da bulur.")
    st.code("""
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)""")
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train)
    st.write("Model Score:",model.score(x_test, y_test))


    st.subheader("Cross-Validation ile Modelin Performansının Ölçülmesi")
    st.write("Cross-Validation veriyi belirlenen kadar parçalayar ve her parçayı ayrı ayrı modelde test eder. Sonuç olarak modelin gerçek performansının ortaya konmasını amaçlar.")
    st.code("cross_val_score(model, x, y, cv=5).mean()")
    lr_cros_val = cross_val_score(model, x, y, cv=5).mean()
    st.write("Modelin Cross-Validation Performansı:",lr_cros_val)


    st.subheader("Modelin Performansının Confusion Matrix Üzerinde Gösterimi")
    st.write("""
             <b>Confusion matrix (karmaşıklık matrisi)</b>, bir sınıflandırma modelinin performansını değerlendirmek için kullanılan bir tablodur. Bu matris, gerçek sınıf etiketleri ile modelin tahmin ettiği sınıf etiketlerini karşılaştırır ve dört temel değeri içerir:

    <b> True Positive (TP - Doğru Pozitif):</b> Modelin doğru bir şekilde pozitif olarak tahmin ettiği durum sayısı.

    <b>True Negative (TN - Doğru Negatif):</b> Modelin doğru bir şekilde negatif olarak tahmin ettiği durum sayısı.

    <b>False Positive (FP - Yanlış Pozitif):</b> Modelin negatif olan bir durumu pozitif olarak yanlış bir şekilde tahmin ettiği durum sayısı.

    <b>False Negative (FN - Yanlış Negatif):</b> Modelin pozitif olan bir durumu negatif olarak yanlış bir şekilde tahmin ettiği durum sayısı.
             """,unsafe_allow_html=True)
    y_predicted = model.predict(x_test)
    cm= confusion_matrix(y_test, y_predicted)
    st.code("""
    y_predicted = model.predict(x_test)
confusion_matrix(y_test, y_predicted)""")
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    st.write("Confusion Matrix:")
    st.write(cm_df)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Perished', 'Survived'])
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.code(classification_report(y_test, y_predicted),language='text')

    st.subheader("Model Performansının ROC Curve Üzerinde Gösterimi")
    st.write("ROC Curve (Receiver Operating Characteristic Curve), bir sınıflandırma modelinin performansını değerlendirmek için kullanılan bir grafiktir. ROC eğrisi, bir modelin duyarlılık (sensitivity) ve özgüllük (specificity) performansını görsel olarak gösterir. Daha spesifik olarak, farklı sınırlama (threshold) değerlerinde elde edilen True Positive Rate (TPR) ve False Positive Rate (FPR) değerlerini çizer.")
    st.code(""" 
y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name='Logistic Regression')
roc_display.plot(ax=ax)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
st.pyplot(fig)""")
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name='Logistic Regression')
    roc_display.plot(ax=ax)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    st.pyplot(fig)


    st.title("Modelin Girilecek Değerler İle Tahminlemesi")


    st.subheader("Bilgi Girişi")

    # 1. Yaş Bilgisi
    age = st.slider("Yaşınızı Seçin:", min_value=1, max_value=100, value=30)

    # 2. Cinsiyet Bilgisi
    gender = st.radio("Cinsiyetinizi Seçin:", ["Erkek", "Kadın"])

    # 3. Pclass Bilgisi
    pclass = st.selectbox("Hangi Pclass'ta Uçuyorsunuz?", [1, 2, 3])

    gender_mapping = {"Erkek": [1, 0], "Kadın": [0, 1]}
    pclass_mapping = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}

    infos = np.concatenate(([age], gender_mapping[gender], pclass_mapping[pclass])).reshape(1, -1)

    # Oluşturulan array'i göster
    st.write("Oluşturulan Array:", (infos))

    cevap = " <b>Hayır</b> :skull:"
    tahmin = model.predict(infos)[0]
    if tahmin:
        cevap = " <b> Evet</b> :sunglasses:"
    st.write("Bu bilgilere sahip bir kişi hayatta kalır mıydı?", cevap,unsafe_allow_html=True)
    probability = model.predict_proba(infos)[0][1]
    st.write(f'Hayatta kalma olasılığı: <b>{probability:.1%}</b>',unsafe_allow_html=True)


with st.expander("Decision Tree Algoritması"):
    st.header("Decision Tree Algoritması ile Sınıflandırma")
    st.write("""
Decision Tree (Karar Ağacı), bir sınıflandırma veya regresyon görevini gerçekleştirmek için kullanılan bir makine öğrenimi algoritmasıdır. Veri kümesini belirli bir hedef değişkeni tahmin etmek veya sınıflandırmak için kullanılır. Ağaç yapısı, veri kümesindeki özelliklere dayanarak bir dizi karar düğümü içerir. Her düğüm, bir özellikle ilişkilidir ve veriyi bu özellik üzerinden bölerek daha küçük alt kümeler oluşturur. Bu bölünmeler, veriyi sınıflandırmak veya regresyon tahminlemek için kullanılır.""")
    
    data = pd.read_csv("titanic.csv")
    # Store the 'Survived' feature in a new variable and remove it from the dataset
    outcomes = data['Survived']
    features_raw = data.drop('Survived', axis = 1)

    # Show the new dataset with 'Survived' removed
    st.dataframe(features_raw.head())
    # Removing the names
    features_no_names = features_raw.drop(['Name'], axis=1)

    # One-hot encoding
    st.subheader("One-Hot Encoding")
    st.write("One-Hot Encoding, kategorik verilerin makine öğrenimi modelleri tarafından daha iyi işlenebilmesi için kullanılan bir dönüşüm yöntemidir. Bu yöntem, kategorik verileri sayısal formata dönüştürerek modelin bu verileri daha iyi anlamasına yardımcı olur.")
    st.code(""" 
features = pd.get_dummies(features_no_names)
features = features.fillna(0.0)
st.dataframe(features.head())""")
    features = pd.get_dummies(features_no_names)
    features = features.fillna(0.0)
    st.dataframe(features.head())
    st.subheader("Modelin Eğitilmesi")
    st.write("Veriyi train ve test olarak iki parçaya ayırdıktan sonra kullanılacak model belirlenir. Bu model üzerinde train veri seti kullanılarak eğitim gerçekleştirilir.")
    st.code(""" 
dtx_train, dtx_test, dty_train, dty_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)
model_dt = DecisionTreeClassifier()
model_dt.fit(dtx_train,dty_train)""")
    
    dtx_train, dtx_test, dty_train, dty_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)
    model_dt = DecisionTreeClassifier()
    model_dt.fit(dtx_train,dty_train)
    
    
    # Making predictions
    dty_train_pred = model_dt.predict(dtx_train)
    dty_test_pred = model_dt.predict(dtx_test)
    train_accuracy = accuracy_score(dty_train, dty_train_pred)
    test_accuracy = accuracy_score(dty_test, dty_test_pred)
    st.write('Modelin Train Veri Setinde Accuracy Değeri', train_accuracy)
    st.write('Modelin Test Veri Setinde Accuracy Değeri', test_accuracy)
    
    st.subheader("Decision Tree Hyperparametre Optimizasyonu")
    # TODO: Train the model
    st.markdown("""
    Decision Tree algoritmasına hyperparametre optimizasyonu uygulanması,
                    
•	**max_depth**: Ağacın en fazla kaç seviyeye kadar büyüyeceğini belirten bir hiperparametredir. Bu, ağacın karmaşıklığını kontrol eder ve aşırı öğrenmeyi önler.

•	**min_samples_leaf**: Bir yaprak düğümünde olması gereken minimum örnek sayısını belirten bir hiperparametredir. Bu, ağacın belirli bir dalga kadar inmesini kontrol eder ve aşırı öğrenmeyi önler.

•	**min_samples_split**: Bir düğümü bölmek için gereken minimum örnek sayısını belirten bir hiperparametredir. Bu, ağacın yeni dallara bölünmesini kontrol eder.""")
    st.code(""" 
model_dtt = DecisionTreeClassifier(max_depth= 10,min_samples_leaf =5, min_samples_split=6)
model_dtt.fit(dtx_train,dty_train)""")
    model_dtt = DecisionTreeClassifier(max_depth= 10,min_samples_leaf =5, min_samples_split=6)
    model_dtt.fit(dtx_train,dty_train)
    # TODO: Make predictions
    dtty_train_pred = model_dtt.predict(dtx_train)
    dtty_test_pred = model_dtt.predict(dtx_test)

    # TODO: Calculate the accuracy
    dttrain_accuracy = accuracy_score(dty_train, dtty_train_pred)
    dttest_accuracy = accuracy_score(dty_test, dtty_test_pred)
    
    st.write('Modelin Test Veri Setinde Hyperparametre Optimzasyonu Sonrası Accuracy Değeri:', dttest_accuracy)
    if st.button("Decision Tree Yapısı"):
        st.write("Modelin ağaç yapısı aşağıdaki gibidir.")
        figg, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model_dt, filled=True, feature_names=features.columns, class_names=['Perished', 'Survived'], ax=ax)
        st.pyplot(figg)


with st.expander("Support Vector Machine"):
    st.header("Support Vector Machine Algoritması ile Sınıflandırma")
    st.markdown("""
Support Vector Machine (SVM), özellikle sınıflandırma görevlerinde kullanılan bir makine öğrenimi algoritmasıdır. Temel amacı, veri noktalarını sınıflandırırken, sınıflar arasındaki en geniş ayrımı (margin) bulmaktır.

SVM'nin bileşenleri şunlardır:

**Margin Maksimizasyonu:** SVM, veri noktalarını sınıflandırırken, sınıflar arasındaki boşluğu (margin) maksimize etmeye çalışır. Margin, karar sınırı ile en yakın veri noktası arasındaki uzaklığı ifade eder.

**Support Vectors**: SVM'nin sınıflandırma sırasında odaklandığı önemli noktalara "destek vektörler" denir. Bu vektörler, margin üzerindeki veya margin sınırları içindeki veri noktalarını temsil eder.

**Kernel**: SVM, doğrusal olmayan veri setlerinde de kullanılabilir. Bunun için kernel fonksiyonları kullanılır. Kernel fonksiyonları, veriyi yüksek boyutlu uzaylara taşıyarak doğrusal olmayan ilişkileri ifade edebilir.

SVM, sınıflandırma yanında regresyon görevlerinde de kullanılabilir. SVM'nin avantajlarından biri, özellikle yüksek boyutlu veri setlerinde etkili olabilmesi ve aşırı uydurmanın önlenmesine yardımcı olmasıdır.
""")
    st.image('svm.webp', caption="""SVM'in Görselleştirilmesi. Margine - Destek Vektörleri """)
    st.subheader("Verinin Modellemeye Hazır Hale Getirilmesi")
    st.write("Kullanılmayacak kolonlar çıkartılıp cinsiyet ve Pclass bilgilerinin lojik hale getirilmesi")
    st.code("""
df_svm = pd.read_csv('titanic.csv')
df_svm = df_svm[['Survived', 'Age', 'Sex', 'Pclass']]
df_svm = pd.get_dummies(df_svm, columns=['Sex', 'Pclass'])
df_svm.dropna(inplace=True)
""")
    df_svm = pd.read_csv('titanic.csv')
    df_svm = df_svm[['Survived', 'Age', 'Sex', 'Pclass']]
    df_svm = pd.get_dummies(df_svm, columns=['Sex', 'Pclass'])
    df_svm.dropna(inplace=True)
    st.write("Veriden Survived kolonu ayrıştırılıyor ve veri eğitim ve test verisi olarak ayrılıyor.")
    st.code("""
x_svm = df_svm.drop('Survived', axis=1)
y_svm = df_svm['Survived']
x_svm_train, x_svm_test, y_svm_train, y_svm_test = train_test_split(x_svm, y_svm, test_size=0.2, stratify=y_svm, random_state=0)""")
    x_svm = df_svm.drop('Survived', axis=1)
    y_svm = df_svm['Survived']

    x_svm_train, x_svm_test, y_svm_train, y_svm_test = train_test_split(x_svm, y_svm, test_size=0.2, stratify=y_svm, random_state=0)
    st.subheader("Modelin Eğitilmesi")
    st.markdown("##### Standart Paramtereler ile Eğitilmesi")
    st.code("""
modelsvm = SVC(probability=True, random_state=0)
modelsvm.fit(x_svm_train, y_svm_train)""")
    modelsvm = SVC(probability=True, random_state=0)
    modelsvm.fit(x_svm_train, y_svm_train)
    st.write("Modelin standart parametreler ile accuracy değeri:",modelsvm.score(x_svm_test, y_svm_test))
    
    st.markdown("##### Lineer Kernel ile Eğitilmesi")
    st.code("""
modelsvmm = SVC(kernel='linear', probability=True, random_state=0)
modelsvmm.fit(x_svm_train, y_svm_train)
""")
    modelsvmm = SVC(kernel='linear', probability=True, random_state=0)
    modelsvmm.fit(x_svm_train, y_svm_train)
    st.write("Modelin lineer kernel ile accuracy değeri:",modelsvmm.score(x_svm_test, y_svm_test))
    
    st.markdown("###### Lineer Kernel ile Eğitilmiş Modelin Cross Validation Performansı")
    st.code("cross_validate(modelsvmm, x_svm, y_svm, cv=5)")
    scoressvm = cross_validate(modelsvmm, x_svm, y_svm, cv=5)
    st.write("Modelin Lineer Kernel ile cross-validation sonrası accuracy değeri ",scoressvm['test_score'].mean())
    
    
    if st.button("Hiperparametre Optimizasyonu - 1"):    
        st.markdown("GridSearch metodu ile belirlediğimiz parametreler aralığında en iyi performans değerini sağlayan modelin bulunması amaçlanır. **C**, **gamma** ve **kernel** değerlerinin kombinasyonu ile 75 farklı model üretilir ve test edilir.")
        st.code("""
modelgrid = SVC(probability=True, random_state=0)
param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'gamma': [0.1, 0.5, 1, 5, 10],
    'kernel': ['linear', 'rbf', 'sigmoid']
}
grid_search = GridSearchCV(estimator=modelgrid, param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(x, y) 
        """)
        modelgrid = SVC(probability=True, random_state=0)

        param_grid = {
            'C': [0.1, 0.5, 1, 5, 10],
            'gamma': [0.1, 0.5, 1, 5, 10],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }
        param_gridd = {
            'C': [0.1],
            'gamma': [0.1],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }
        grid_search = GridSearchCV(estimator=modelgrid, param_grid=param_grid, cv=5, verbose=2)
        grid_search.fit(x, y) # Train the model 75 times with 75 different parameter combinations

        best_model = grid_search.best_estimator_ 
        scores_best_model = cross_validate(best_model, x, y, cv=5)
        st.write("Bu parametreler arasında bulunan en iyi modelin parametreleri:",grid_search.best_params_)
        st.write("Bu parametreler arasında bulunan en iyi modelin accuracy değeri",scores_best_model['test_score'].mean())
        
    if st.button("Hiperparametre Optimizasyonu - 2"):   
        st.write("Kernel değeri **rbf** olarak belirlenip diğer parametrelerin karşılaştırılarak en iyi modelin bulunması.")
        st.code(""" 
modelgrid2 = SVC(kernel='rbf', probability=True, random_state=0)
param_grid2 = {
    'C': [1, 2, 3],
    'gamma': [0.25, 0.5, 0.75]
}
grid_search2 = GridSearchCV(estimator=modelgrid2, param_grid=param_grid2, cv=5, verbose=2)
grid_search2.fit(x, y)        
        """)
        modelgrid2 = SVC(kernel='rbf', probability=True, random_state=0)
        param_grid2 = {
            'C': [1, 2, 3],
            'gamma': [0.25, 0.5, 0.75]
        }
        grid_search2 = GridSearchCV(estimator=modelgrid2, param_grid=param_grid2, cv=5, verbose=2)
        grid_search2.fit(x, y)

        best_model2 = grid_search2.best_estimator_
        st.write("Bu parametreler arasında bulunan en iyi modelin parametreleri:",grid_search2.best_params_)
        scores2 = cross_validate(best_model2, x, y, cv=5)
        st.write("Bu parametreler arasında bulunan en iyi modelin accuracy değeri",scores2['test_score'].mean())
        
    st.subheader("Modelin Girilecek Değerler İle Tahminlemesi")


    st.markdown("##### Bilgi Girişi")

    # 1. Yaş Bilgisi
    agesvm = st.slider("Yaşınız:", min_value=1, max_value=100, value=30)

    # 2. Cinsiyet Bilgisi
    gendersvm = st.radio("Cinsiyetiniz:", ["Erkek", "Kadın"])

    # 3. Pclass Bilgisi
    pclasssvm = st.selectbox("Hangi Pclass'ta Uçuyorsunuz? ", [1, 2, 3])

    gender_mappingsvm = {"Erkek": [1, 0], "Kadın": [0, 1]}
    pclass_mappingsvm = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}

    infosvm = np.concatenate(([agesvm], gender_mappingsvm[gendersvm], pclass_mappingsvm[pclasssvm])).reshape(1, -1)

    # Oluşturulan array'i göster
    st.write("Oluşturulan Array:", (infosvm))

    cevapsvm = " <b>Hayır</b> :skull:"
    tahminsvm = modelsvmm.predict(infosvm)[0]
    if tahminsvm:
        cevapsvm = " <b> Evet</b> :sunglasses:"
    st.write("Bu bilgilere sahip bir kişi hayatta kalır mıydı?", cevapsvm,unsafe_allow_html=True)
    probabilitysvm = modelsvmm.predict_proba(infosvm)[0][1]
    st.write(f'Hayatta kalma olasılığı: <b>{probabilitysvm:.1%}</b>',unsafe_allow_html=True)

with st.expander("Sonuç"):
    st.markdown("## Yapılan Çalışmaların Sonuçlandırılması")
    st.markdown(f""" 
    ##### • Modellerin Karşılaştırılması:
Cross-Validation sonuçlarına göre Decision Tree algoritması {dttest_accuracy:.1%} ile en yüksek accucarcy değerine sahip olduğu gözlenmiştir. Lojistik Regresyon ve SVM algoritmalarının accucarcy değerleri sırasıyla {lr_cros_val:.1%} ile {scoressvm['test_score'].mean():.1%} değerlerine sahiptir. \n
""")
    
    st.markdown(f""" 
    ##### • Duyarlılık Analizi:
    Model performansının belirli hiperparametrelerle değişimi değerlendirildi. Lojistik regresyon modeli, C parametresindeki değişikliklere duyarlılık gösterirken, decision tree modeli en çok max_depth parametresine, SVM modeli ise gamma parametresine duyarlılık gösterdi.
    """)
    
    st.markdown(f""" 
    ##### • Modellerin Karakteristik Yaklaşımları:
    Her bir modelin kendine özgü karakteristik yaklaşımları bulunmaktadır. Lojistik regresyon, lineer ilişkileri öğrenme yeteneği ile öne çıkarken, decision tree modeli veri setindeki karmaşıklıkları daha iyi yakalayabilir. SVM ise yapısı gereği karar destek vektörlerini esnek bir şekilde ayarlayabilmesi kullanım avantajı sağlar.
    """)
    

    
    
    
    
    
    
    
    
