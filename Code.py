# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:07:44 2018

@author: ma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, recall_score, precision_score,f1_score, precision_recall_curve

Combined = pd.read_csv('xxx/xxx.csv')

Combined.set_index('OrderID',inplace=True)

Combined_dummies = pd.get_dummies(Combined,columns=['Symbol','VenueType','Side','SecurityCategory','Sector','MktCap'])

if 'TradeDateTime' in Combined_dummies.columns:
    print('yes')
    
Combined_dummies['StartTime']=pd.to_datetime(Combined_dummies['TradeDateTime'])

quantile = Combined_dummies['StartTime'].quantile(0.5)

combined_training = Combined_dummies[Combined_dummies['StartTime']<=quantile]
combined_testing  = Combined_dummies[Combined_dummies['StartTime']>quantile]

del quantile

training_trades = combined_training.groupby('BecomeTrade').BecomeTrade.count()[1]
testing_trades = combined_testing.groupby('BecomeTrade').BecomeTrade.count()[1] 

combined_training.fillna(0, inplace = True)
combined_testing.fillna(0, inplace = True)

normalizationNames = [
 'Adv20d',
 'L1_1',                    
 'L1_5',                     
 'L1_15',                    
 'L1_Open',                  
 'L1_Previous',             
 'L3_1',                     
 'L3_5',                    
 'L3_15',                   
 'L3_Open',                  
 'L3_Previous'
]

polyFeaturesBase = normalizationNames

unormalizedNames = [
'ContinuousStartTime',
'StartTime',
'Symbol_C00001383',
'Symbol_C00001391',
'Symbol_C00001425',
'Symbol_C00001678',
'Symbol_C00002121',
'Symbol_C00002992',
'Symbol_C00003145',
'Symbol_C00003164',
'Symbol_C00003398',
'Symbol_C00003535',
'Symbol_C00003863',
'Symbol_C00003877',
'Symbol_C00007239',
'Symbol_C00007999',
'Symbol_C00008383',
'Symbol_C00008922',
'Symbol_C00009433',
'Symbol_C00009561',
'Symbol_C00011235',
'Symbol_C00011255',
'Symbol_C00011384',
'Symbol_C00011416',
'Symbol_C00011459',
'Symbol_C00011735',
'Symbol_C00011754',
'Symbol_C0001182145',
'Symbol_C00012534',
'Symbol_C00012613',
'Symbol_C00012694',
'Symbol_C00012736',
'Symbol_C00013125',
'Symbol_C00013562',
'Symbol_C00013844',
'Symbol_C00013943',
'Symbol_C00013954',
'Symbol_C00014694',
'Symbol_C00014719',
'Symbol_C00015385',
'Symbol_C00015416',
'Symbol_C00015714',
'Symbol_C00015724',
'Symbol_C00015978',
'Symbol_C00016289',
'Symbol_C00016352',
'Symbol_C00016369',
'Symbol_C00017194',
'Symbol_C00018194',
'Symbol_C00018214',
'Symbol_C00018559',
'Symbol_C00018997',
'Symbol_C0002085971',
'Symbol_C00021112137682',
'Symbol_C00021222',
'Symbol_C00022125',
'Symbol_C00023215',
'Symbol_C00024394',
'Symbol_C00024716',
'Symbol_C00024914',
'Symbol_C00025215',
'Symbol_C00025490',
'Symbol_C00025612',
'Symbol_C00025780',
'Symbol_C00026451',
'Symbol_C00026692',
'Symbol_C00028120',
'Symbol_C0010149931',
'Symbol_C00101699',
'Symbol_C00101743',
'Symbol_C00102143',
'Symbol_C00102199',
'Symbol_C00102499',
'Symbol_C00102953',
'Symbol_C00103115',
'Symbol_C00103124',
'Symbol_C00103181',
'Symbol_C00103182',
'Symbol_C00103255',
'Symbol_C00103359',
'Symbol_C00103441',
'Symbol_C00103527',
'Symbol_C00103815',
'Symbol_C00103946',
'Symbol_C00106147',
'Symbol_C00107924',
'Symbol_C00108199',
'Symbol_C00108397',
'Symbol_C00108447',
'Symbol_C00109457',
'Symbol_C00109537',
'Symbol_C00111162',
'Symbol_C00111389',
'Symbol_C00111436',
'Symbol_C00111439',
'Symbol_C00111485',
'Symbol_C00111491',
'Symbol_C00112952',
'Symbol_C00112959',
'Symbol_C00113189',
'Symbol_C00113336',
'Symbol_C00113384',
'Symbol_C00113449',
'Symbol_C00113526',
'Symbol_C00113584',
'Symbol_C00113639',
'Symbol_C00113816',
'Symbol_C00113848',
'Symbol_C00113896',
'Symbol_C00114565',
'Symbol_C00115353',
'Symbol_C00116488',
'Symbol_C00117189',
'Symbol_C00117536',
'Symbol_C00117935',
'Symbol_C00118125',
'Symbol_C00118326',
'Symbol_C00118494',
'Symbol_C0012017371',
'Symbol_C00121103227685',
'Symbol_C00121103487682',
'Symbol_C00121107767681',
'Symbol_C00121784',
'Symbol_C00122285',
'Symbol_C00122384',
'Symbol_C00123195',
'Symbol_C00123264',
'Symbol_C00123744',
'Symbol_C00123883',
'Symbol_C00125784',
'Symbol_C00126190',
'Symbol_C00127450',
'Symbol_C00128934',
'Symbol_C00129180',
'Symbol_C00203349',
'Symbol_C00203521',
'Symbol_C00203922',
'Symbol_C00206105',
'Symbol_C00206555',
'Symbol_C00206902',
'Symbol_C00206925',
'Symbol_C00209319',
'Symbol_C00209409',
'Symbol_C002110197682',
'Symbol_C00211219',
'Symbol_C00211343',
'Symbol_C00211419',
'Symbol_C00212508',
'Symbol_C00213103',
'Symbol_C00213105',
'Symbol_C00213123',
'Symbol_C00214433',
'Symbol_C00216924',
'Symbol_C00217525',
'Symbol_C00218131',
'Symbol_C00223815',
'Symbol_C00226966',
'Symbol_C00229444',
'Symbol_C01001131',
'Symbol_C01001317',
'Symbol_C01001321',
'Symbol_C01001341',
'Symbol_C01001354',
'Symbol_C01001394',
'Symbol_C01001682',
'Symbol_C01001838',
'Symbol_C01002224',
'Symbol_C01002338',
'Symbol_C01002454',
'Symbol_C01002525',
'Symbol_C01002569',
'Symbol_C01003611',
'Symbol_C01005893',
'Symbol_C01005895',
'Symbol_C01006293',
'Symbol_C01006813',
'Symbol_C01006854',
'Symbol_C01006879',
'Symbol_C01008993',
'Symbol_C01009666',
'Symbol_C01009689',
'Symbol_C0101033391',
'Symbol_C01011113',
'Symbol_C01011115',
'Symbol_C01011225',
'Symbol_C01011274',
'Symbol_C01011374',
'Symbol_C01011376',
'Symbol_C01011475',
'Symbol_C01011496',
'Symbol_C01011613',
'Symbol_C01011644',
'Symbol_C01011843',
'Symbol_C01011879',
'Symbol_C01011974',
'Symbol_C01011988',
'Symbol_C01012169',
'Symbol_C01012233',
'Symbol_C01013248',
'Symbol_C01013588',
'Symbol_C01013899',
'Symbol_C01013997',
'Symbol_C01015359',
'Symbol_C01015446',
'Symbol_C01015613',
'Symbol_C01015721',
'Symbol_C01016243',
'Symbol_C01016298',
'Symbol_C01016474',
'Symbol_C01016814',
'Symbol_C01017229',
'Symbol_C01017575',
'Symbol_C01017869',
'Symbol_C01019374',
'Symbol_C01019421',
'Symbol_C01019438',
'Symbol_C01019464',
'Symbol_C01019465',
'Symbol_C01019843',
'Symbol_C01021101387684',
'Symbol_C01021101387685',
'Symbol_C01021106837686',
'Symbol_C01021254',
'Symbol_C01021310',
'Symbol_C01021334',
'Symbol_C01021434',
'Symbol_C01021450',
'Symbol_C01021870',
'Symbol_C01023240',
'Symbol_C01023274',
'Symbol_C01023410',
'Symbol_C01023440',
'Symbol_C01023694',
'Symbol_C01025274',
'Symbol_C01025615',
'Symbol_C01025794',
'Symbol_C01026254',
'Symbol_C01026434',
'Symbol_C01027990',
'Symbol_C01029435',
'Symbol_C01029445',
'Symbol_C01029465',
'Symbol_C01029894',
'Symbol_C01101225',
'Symbol_C01101342',
'Symbol_C01101365',
'Symbol_C01101389',
'Symbol_C01101657',
'Symbol_C01101666',
'Symbol_C01101841',
'Symbol_C01101843',
'Symbol_C01101941',
'Symbol_C01102089',
'Symbol_C01102212',
'Symbol_C01102501',
'Symbol_C01102544',
'Symbol_C01103585',
'Symbol_C01103834',
'Symbol_C01103835',
'Symbol_C01103981',
'Symbol_C01104383',
'Symbol_C01104915',
'Symbol_C01105229',
'Symbol_C01106524',
'Symbol_C01107247',
'Symbol_C01107431',
'Symbol_C01107483',
'Symbol_C01107557',
'Symbol_C01109987',
'Symbol_C01111219',
'Symbol_C0111126563',
'Symbol_C01111283',
'Symbol_C01111284',
'Symbol_C01111294',
'Symbol_C01111664',
'Symbol_C01111684',
'Symbol_C01111831',
'Symbol_C01111889',
'Symbol_C0111211035887681',
'Symbol_C01112234',
'Symbol_C01112384',
'Symbol_C01112656',
'Symbol_C01112818',
'Symbol_C01112899',
'Symbol_C01113249',
'Symbol_C01113329',
'Symbol_C01113492',
'Symbol_C01113522',
'Symbol_C01113533',
'Symbol_C01113544',
'Symbol_C01114414',
'Symbol_C01114625',
'Symbol_C01115318',
'Symbol_C01115480',
'Symbol_C01115859',
'Symbol_C01115968',
'Symbol_C01116498',
'Symbol_C01116588',
'Symbol_C01116592',
'Symbol_C01116928',
'Symbol_C01117159',
'Symbol_C01117431',
'Symbol_C01117559',
'Symbol_C01117658',
'Symbol_C01117834',
'Symbol_C01119234',
'Symbol_C01119549',
'Symbol_C01121103587681',
'Symbol_C01121104287689',
'Symbol_C01121112327682',
'Symbol_C01121184',
'Symbol_C01121225',
'Symbol_C01121245',
'Symbol_C01121250',
'Symbol_C01121885',
'Symbol_C0112211012257681',
'Symbol_C01122284',
'Symbol_C01122540',
'Symbol_C01122634',
'Symbol_C01123444',
'Symbol_C01123590',
'Symbol_C01123680',
'Symbol_C01123684',
'Symbol_C01123920',
'Symbol_C01124453',
'Symbol_C01124884',
'Symbol_C01125492',
'Symbol_C01125984',
'Symbol_C01126860',
'Symbol_C01127253',
'Symbol_C01127480',
'Symbol_C01128345',
'Symbol_C01128395',
'Symbol_C01128524',
'Symbol_C01129335',
'Symbol_C01129362',
'Symbol_C01129452',
'Symbol_C01201304',
'Symbol_C01201361',
'Symbol_C01202215',
'Symbol_C01202403',
'Symbol_C01202608',
'Symbol_C01204208',
'Symbol_C01204839',
'Symbol_C01205401',
'Symbol_C01205407',
'Symbol_C01206541',
'Symbol_C01206546',
'Symbol_C01207812',
'Symbol_C01209403',
'Symbol_C01209421',
'Symbol_C01209834',
'Symbol_C012110797681',
'Symbol_C012110797682',
'Symbol_C01211244',
'Symbol_C01211364',
'Symbol_C01211403',
'Symbol_C01211609',
'Symbol_C01213229',
'Symbol_C01213534',
'Symbol_C01213804',
'Symbol_C01213865',
'Symbol_C01214208',
'Symbol_C01215402',
'Symbol_C01215663',
'Symbol_C01216244',
'Symbol_C01217419',
'Symbol_C01218903',
'Symbol_C01219638',
'Symbol_C01219642',
'Symbol_C012210737392',
'Symbol_C01221104',
'Symbol_C01223505',
'Symbol_C01225220',
'Symbol_C01227404',
'Symbol_C01227405',
'Symbol_C01228804',
'Symbol_C01229401',
'Symbol_C01229412',
'Symbol_C02001075',
'Symbol_C02001294',
'Symbol_C02002166',
'Symbol_C02002367',
'Symbol_C02003044',
'Symbol_C02003125',
'Symbol_C02003129',
'Symbol_C02003141',
'Symbol_C02003279',
'Symbol_C02003525',
'Symbol_C02004338',
'Symbol_C02005063',
'Symbol_C02005283',
'Symbol_C02005323',
'Symbol_C02006019',
'Symbol_C02009039',
'Symbol_C02011054',
'Symbol_C02011084',
'Symbol_C02011242',
'Symbol_C02011275',
'Symbol_C02011278',
'Symbol_C02013019',
'Symbol_C02013255',
'Symbol_C02014433',
'Symbol_C02015229',
'Symbol_C02015419',
'Symbol_C02015452',
'Symbol_C02018073',
'Symbol_C02018129',
'Symbol_C02019051',
'Symbol_C02019057',
'Symbol_C02019078',
'Symbol_C02021084',
'Symbol_C02021444',
'Symbol_C02024214',
'Symbol_C02028024',
'Symbol_C02028080',
'Symbol_C02101081',
'Symbol_C02101083',
'Symbol_C02101089',
'Symbol_C02101097',
'Symbol_C02102324',
'Symbol_C02102599',
'Symbol_C02103085',
'Symbol_C02103098',
'Symbol_C02103241',
'Symbol_C02105465',
'Symbol_C02109089',
'Symbol_C02109197',
'Symbol_C02111018',
'Symbol_C02111454',
'Symbol_C0211163541',
'Symbol_C02112182',
'Symbol_C02113084',
'Symbol_C02113086',
'Symbol_C02115429',
'Symbol_C02115465',
'Symbol_C02115468',
'Symbol_C02117029',
'Symbol_C02118646',
'Symbol_C02121102467682',
'Symbol_C02122430',
'Symbol_C02123020',
'Symbol_C02123034',
'Symbol_C02123220',
'Symbol_C02123390',
'Symbol_C02124190',
'Symbol_C02124230',
'Symbol_C02125095',
'Symbol_C02126040',
'Symbol_C02201029',
'Symbol_C02201401',
'Symbol_C02203209',
'Symbol_C02209012',
'Symbol_C022110807683',
'Symbol_C022110807685',
'Symbol_C02212614',
'Symbol_C02213501',
'Symbol_C02213508',
'Symbol_C02214408',
'Symbol_C02215408',
'Symbol_C02219459',
'Symbol_C02222340',
'Symbol_C02229205',
'Symbol_C10002399',
'Symbol_C1000239993',
'Symbol_C10002798',
'Symbol_C10003734',
'Symbol_C10003944',
'Symbol_C10004566',
'Symbol_C10004783',
'Symbol_C10004935',
'Symbol_C10004984',
'Symbol_C10006423',
'Symbol_C10006435',
'Symbol_C10006673',
'Symbol_C10008135',
'Symbol_C10008399',
'Symbol_C10008935',
'Symbol_C10009193',
'Symbol_C10009213',
'Symbol_C10009513',
'Symbol_C10009554',
'Symbol_C10010131',
'Symbol_C10010316',
'Symbol_C10013386',
'Symbol_C10013546',
'Symbol_C10013596',
'Symbol_C10016435',
'Symbol_C10017452',
'Symbol_C10017754',
'Symbol_C10018192',
'Symbol_C10018464',
'Symbol_C10018534',
'Symbol_C10018574',
'Symbol_C10018754',
'Symbol_C10018972',
'Symbol_C10019128',
'Symbol_C10019519',
'Symbol_C10019754',
'Symbol_C10019811',
'Symbol_C10019954',
'Symbol_C10021104557683',
'Symbol_C10021106527683',
'Symbol_C10021106597681',
'Symbol_C10021106597682',
'Symbol_C10021106597683',
'Symbol_C10023384',
'Symbol_C10023550',
'Symbol_C10024252',
'Symbol_C10026133',
'Symbol_C10029154',
'Symbol_C10029383',
'Symbol_C10029384',
'Symbol_C10101709',
'Symbol_C1010192816',
'Symbol_C10103148',
'Symbol_C10103181',
'Symbol_C10103769',
'Symbol_C10103824',
'Symbol_C10104328',
'Symbol_C10104339',
'Symbol_C10104557',
'Symbol_C10106262',
'Symbol_C10108185',
'Symbol_C10108568',
'Symbol_C10109144',
'Symbol_C10109268',
'Symbol_C10109989',
'Symbol_C10110195',
'Symbol_C10110353',
'Symbol_C10110922',
'Symbol_C10112954',
'Symbol_C10113427',
'Symbol_C10113495',
'Symbol_C10114393',
'Symbol_C10115328',
'Symbol_C10115886',
'Symbol_C10116544',
'Symbol_C10116989',
'Symbol_C10117353',
'Symbol_C10118522',
'Symbol_C10119198',
'Symbol_C10119482',
'Symbol_C10119549',
'Symbol_C10119735',
'Symbol_C10119739',
'Symbol_C10119856',
'Symbol_C10119858',
'Symbol_C10120221',
'Symbol_C1012084971',
'Symbol_C1012084972',
'Symbol_C10121101937689',
'Symbol_C10121102857689',
'Symbol_C10121108197681',
'Symbol_C10121109147689',
'Symbol_C10121109857685',
'Symbol_C10121113587681',
'Symbol_C10121584',
'Symbol_C10122941',
'Symbol_C10122970',
'Symbol_C10123225',
'Symbol_C10123250',
'Symbol_C10123424',
'Symbol_C10123426',
'Symbol_C10123484',
'Symbol_C10123744',
'Symbol_C10123940',
'Symbol_C10124624',
'Symbol_C10125594',
'Symbol_C10126143',
'Symbol_C10126384',
'Symbol_C10126660',
'Symbol_C10126693',
'Symbol_C10201105',
'Symbol_C10203447',
'Symbol_C10204108',
'Symbol_C10204109',
'Symbol_C10204122',
'Symbol_C10204129',
'Symbol_C10206259',
'Symbol_C10207939',
'Symbol_C10209125',
'Symbol_C10209905',
'Symbol_C10209922',
'Symbol_C102110057681',
'Symbol_C10211559',
'Symbol_C1021211085487681',
'Symbol_C10212118',
'Symbol_C10212139',
'Symbol_C10212619',
'Symbol_C10212924',
'Symbol_C10213901',
'Symbol_C10213902',
'Symbol_C10214538',
'Symbol_C10216507',
'Symbol_C10219756',
'Symbol_C1022022051',
'Symbol_C1022122051',
'Symbol_C10221553',
'Symbol_C10223211',
'Symbol_C10226154',
'Symbol_C10226305',
'Symbol_C10226504',
'Symbol_C10226945',
'Symbol_C10228544',
'Symbol_C10229214',
'Symbol_C11000321',
'Symbol_C11000435',
'Symbol_C11001213',
'Symbol_C11001293',
'Symbol_C1100211086197681',
'Symbol_C11003567',
'Symbol_C11003833',
'Symbol_C11003949',
'Symbol_C11003967',
'Symbol_C11005823',
'Symbol_C11006544',
'Symbol_C11006811',
'Symbol_C11008141',
'Symbol_C11009231',
'Symbol_C11009339',
'Symbol_C11009666',
'Symbol_C1101198326',
'Symbol_C11012164',
'Symbol_C11012573',
'Symbol_C11013235',
'Symbol_C11013544',
'Symbol_C11013974',
'Symbol_C11015674',
'Symbol_C11015832',
'Symbol_C11015999',
'Symbol_C11016466',
'Symbol_C11016683',
'Symbol_C11016836',
'Symbol_C11016875',
'Symbol_C11019274',
'Symbol_C11019416',
'Symbol_C11019614',
'Symbol_C11019832',
'Symbol_C11020185',
'Symbol_C11021101837688',
'Symbol_C11021106917682',
'Symbol_C11021106917683',
'Symbol_C11021106917684',
'Symbol_C11021106917685',
'Symbol_C11021126917683',
'Symbol_C11021126927680',
'Symbol_C11021126927681',
'Symbol_C11021126927682',
'Symbol_C11021126927683',
'Symbol_C11022254',
'Symbol_C11022740',
'Symbol_C11022834',
'Symbol_C11023835',
'Symbol_C11023960',
'Symbol_C11025680',
'Symbol_C11026215',
'Symbol_C11028275',
'Symbol_C11029444',
'Symbol_C11102063',
'Symbol_C11102621',
'Symbol_C11103414',
'Symbol_C11105229',
'Symbol_C11105886',
'Symbol_C11106295',
'Symbol_C11106481',
'Symbol_C11106822',
'Symbol_C11109501',
'Symbol_C11109645',
'Symbol_C11109669',
'Symbol_C11109881',
'Symbol_C11109943',
'Symbol_C11109949',
'Symbol_C11110468',
'Symbol_C11111368',
'Symbol_C11111492',
'Symbol_C11111589',
'Symbol_C1111292453',
'Symbol_C11113485',
'Symbol_C11113582',
'Symbol_C11113824',
'Symbol_C11114241',
'Symbol_C11114294',
'Symbol_C11116263',
'Symbol_C11117064',
'Symbol_C11118522',
'Symbol_C11118966',
'Symbol_C11119245',
'Symbol_C11119549',
'Symbol_C11119582',
'Symbol_C11119621',
'Symbol_C11119749',
'Symbol_C11121103487683',
'Symbol_C11121104447686',
'Symbol_C11121840',
'Symbol_C11122344',
'Symbol_C11123490',
'Symbol_C11125825',
'Symbol_C11126240',
'Symbol_C11126590',
'Symbol_C11126833',
'Symbol_C11128490',
'Symbol_C11128590',
'Symbol_C11129860',
'Symbol_C11202608',
'Symbol_C11203401',
'Symbol_C11204907',
'Symbol_C11205101',
'Symbol_C11206539',
'Symbol_C11206801',
'Symbol_C11206907',
'Symbol_C11208309',
'Symbol_C11209638',
'Symbol_C11210644',
'Symbol_C1121298141',
'Symbol_C11213638',
'Symbol_C11213822',
'Symbol_C11214108',
'Symbol_C11214604',
'Symbol_C11216842',
'Symbol_C11218959',
'Symbol_C11219901',
'Symbol_C11219959',
'Symbol_C11221109937688',
'Symbol_C11223350',
'Symbol_C11228550',
'Symbol_C11229544',
'Symbol_C12001352',
'Symbol_C1200211090177682',
'Symbol_C1200211090177683',
'Symbol_C1200230623',
'Symbol_C12003038',
'Symbol_C12004053',
'Symbol_C12004241',
'Symbol_C12004532',
'Symbol_C12005093',
'Symbol_C12007114',
'Symbol_C12007231',
'Symbol_C12008129',
'Symbol_C12009011',
'Symbol_C12012468',
'Symbol_C12013493',
'Symbol_C12013574',
'Symbol_C12014059',
'Symbol_C12014096',
'Symbol_C12014114',
'Symbol_C12014238',
'Symbol_C12014247',
'Symbol_C12016054',
'Symbol_C12017191',
'Symbol_C12021109197681',
'Symbol_C12022315',
'Symbol_C12024292',
'Symbol_C12026030',
'Symbol_C12029360',
'Symbol_C12029520',
'Symbol_C12103042',
'Symbol_C12103099',
'Symbol_C12103125',
'Symbol_C12103191',
'Symbol_C12103587',
'Symbol_C12104081',
'Symbol_C12104089',
'Symbol_C12104469',
'Symbol_C12106021',
'Symbol_C12107041',
'Symbol_C12107185',
'Symbol_C12109533',
'Symbol_C12113029',
'Symbol_C12114089',
'Symbol_C12114229',
'Symbol_C12115681',
'Symbol_C12116136',
'Symbol_C12116562',
'Symbol_C12119012',
'Symbol_C12119036',
'Symbol_C12119164',
'Symbol_C12119319',
'Symbol_C12122121',
'Symbol_C12122484',
'Symbol_C12124044',
'Symbol_C12124490',
'Symbol_C12124580',
'Symbol_C12127150',
'Symbol_C12128084',
'Symbol_C12200371',
'Symbol_C1220162026',
'Symbol_C12204121',
'Symbol_C12206202',
'Symbol_C12211035',
'Symbol_C12213432',
'Symbol_C12214403',
'Symbol_C12217035',
'Symbol_C20000332',
'Symbol_C2000109332',
'Symbol_C20001329',
'Symbol_C20001363',
'Symbol_C20002912',
'Symbol_C20003969',
'Symbol_C20007113',
'Symbol_C20007127',
'Symbol_C20007149',
'Symbol_C20007157',
'Symbol_C20007213',
'Symbol_C20007218',
'Symbol_C20007272',
'Symbol_C20007289',
'Symbol_C20007317',
'Symbol_C20007318',
'Symbol_C20007319',
'Symbol_C20007322',
'Symbol_C20007327',
'Symbol_C20007333',
'Symbol_C20007351',
'Symbol_C20007367',
'Symbol_C20007384',
'Symbol_C20007385',
'Symbol_C20007413',
'Symbol_C20007424',
'Symbol_C20007439',
'Symbol_C20007443',
'Symbol_C20007444',
'Symbol_C20007459',
'Symbol_C20007514',
'Symbol_C20007546',
'Symbol_C20007561',
'Symbol_C20007567',
'Symbol_C20007623',
'Symbol_C20007643',
'Symbol_C20007697',
'Symbol_C20007699',
'Symbol_C20007712',
'Symbol_C20007739',
'Symbol_C20007749',
'Symbol_C20007753',
'Symbol_C20007772',
'Symbol_C20007777',
'Symbol_C20007829',
'Symbol_C20007831',
'Symbol_C20007833',
'Symbol_C20007842',
'Symbol_C20007852',
'Symbol_C20007897',
'Symbol_C20007913',
'Symbol_C20007935',
'Symbol_C20007986',
'Symbol_C20010513',
'Symbol_C20012352',
'Symbol_C20012535',
'Symbol_C20012932',
'Symbol_C20014353',
'Symbol_C20017112',
'Symbol_C20017116',
'Symbol_C20017134',
'Symbol_C20017143',
'Symbol_C20017146',
'Symbol_C20017149',
'Symbol_C20017153',
'Symbol_C20017155',
'Symbol_C20017162',
'Symbol_C20017174',
'Symbol_C20017178',
'Symbol_C20017213',
'Symbol_C20017235',
'Symbol_C20017238',
'Symbol_C20017244',
'Symbol_C20017254',
'Symbol_C20017279',
'Symbol_C20017295',
'Symbol_C20017296',
'Symbol_C20017318',
'Symbol_C20017329',
'Symbol_C20017334',
'Symbol_C20017353',
'Symbol_C20017368',
'Symbol_C20017389',
'Symbol_C20017412',
'Symbol_C20017418',
'Symbol_C20017435',
'Symbol_C20017448',
'Symbol_C20017449',
'Symbol_C20017469',
'Symbol_C20017488',
'Symbol_C20017499',
'Symbol_C20017529',
'Symbol_C20017532',
'Symbol_C20017548',
'Symbol_C20017553',
'Symbol_C20017556',
'Symbol_C20017572',
'Symbol_C20017623',
'Symbol_C20017634',
'Symbol_C20017649',
'Symbol_C20017653',
'Symbol_C20017654',
'Symbol_C20017672',
'Symbol_C20017699',
'Symbol_C20017753',
'Symbol_C20017758',
'Symbol_C20017776',
'Symbol_C20017793',
'Symbol_C20017799',
'Symbol_C20017812',
'Symbol_C20017819',
'Symbol_C20017836',
'Symbol_C20017849',
'Symbol_C20017859',
'Symbol_C20017895',
'Symbol_C20017923',
'Symbol_C20022390',
'Symbol_C20022552',
'Symbol_C20022912',
'Symbol_C20023415',
'Symbol_C20024290',
'Symbol_C20027120',
'Symbol_C20027125',
'Symbol_C20027150',
'Symbol_C20027190',
'Symbol_C20027192',
'Symbol_C20027214',
'Symbol_C20027220',
'Symbol_C20027244',
'Symbol_C20027310',
'Symbol_C20027364',
'Symbol_C20027460',
'Symbol_C20027485',
'Symbol_C20027492',
'Symbol_C20027510',
'Symbol_C20027554',
'Symbol_C20027594',
'Symbol_C20027634',
'Symbol_C20027690',
'Symbol_C20027692',
'Symbol_C20027720',
'Symbol_C20027724',
'Symbol_C20027893',
'Symbol_C20027954',
'Symbol_C20100334',
'Symbol_C20100741',
'Symbol_C20100824',
'Symbol_C20100935',
'Symbol_C20101299',
'Symbol_C20102125',
'Symbol_C20102381',
'Symbol_C20102499',
'Symbol_C20103947',
'Symbol_C20104381',
'Symbol_C20106281',
'Symbol_C20107107',
'Symbol_C20107117',
'Symbol_C20107122',
'Symbol_C20107134',
'Symbol_C20107135',
'Symbol_C20107137',
'Symbol_C20107138',
'Symbol_C20107146',
'Symbol_C20107159',
'Symbol_C20107161',
'Symbol_C20107163',
'Symbol_C20107164',
'Symbol_C20107168',
'Symbol_C20107184',
'Symbol_C20107193',
'Symbol_C20107322',
'Symbol_C20107326',
'Symbol_C20107329',
'Symbol_C20107331',
'Symbol_C20107333',
'Symbol_C20107334',
'Symbol_C20107335',
'Symbol_C20107337',
'Symbol_C20107339',
'Symbol_C20107341',
'Symbol_C20107343',
'Symbol_C20107356',
'Symbol_C20107357',
'Symbol_C20107362',
'Symbol_C20107365',
'Symbol_C20107383',
'Symbol_C20107386',
'Symbol_C20107389',
'Symbol_C20107421',
'Symbol_C20107422',
'Symbol_C20107442',
'Symbol_C20107453',
'Symbol_C20107526',
'Symbol_C20107534',
'Symbol_C20107554',
'Symbol_C20107557',
'Symbol_C20107564',
'Symbol_C20107565',
'Symbol_C20107633',
'Symbol_C20107639',
'Symbol_C20107644',
'Symbol_C20107646',
'Symbol_C20107683',
'Symbol_C20107684',
'Symbol_C20107721',
'Symbol_C20107763',
'Symbol_C20107826',
'Symbol_C20107849',
'Symbol_C20107857',
'Symbol_C20107887',
'Symbol_C20107926',
'Symbol_C20107967',
'Symbol_C20107994',
'Symbol_C20110522',
'Symbol_C20110586',
'Symbol_C20114219',
'Symbol_C20115526',
'Symbol_C20116954',
'Symbol_C20116956',
'Symbol_C20117119',
'Symbol_C20117122',
'Symbol_C20117125',
'Symbol_C20117134',
'Symbol_C20117136',
'Symbol_C20117154',
'Symbol_C20117159',
'Symbol_C20117221',
'Symbol_C20117222',
'Symbol_C20117239',
'Symbol_C20117323',
'Symbol_C20117328',
'Symbol_C20117336',
'Symbol_C20117339',
'Symbol_C20117341',
'Symbol_C20117345',
'Symbol_C20117347',
'Symbol_C20117352',
'Symbol_C20117355',
'Symbol_C20117356',
'Symbol_C20117358',
'Symbol_C20117383',
'Symbol_C20117419',
'Symbol_C20117469',
'Symbol_C20117492',
'Symbol_C20117529',
'Symbol_C20117538',
'Symbol_C20117549',
'Symbol_C20117578',
'Symbol_C20117625',
'Symbol_C20117628',
'Symbol_C20117629',
'Symbol_C20117658',
'Symbol_C20117662',
'Symbol_C20117699',
'Symbol_C20117735',
'Symbol_C20117742',
'Symbol_C20117764',
'Symbol_C20117829',
'Symbol_C20117854',
'Symbol_C20117860',
'Symbol_C20117867',
'Symbol_C20117882',
'Symbol_C20117908',
'Symbol_C20117945',
'Symbol_C20117983',
'Symbol_C20117988',
'Symbol_C20120544',
'Symbol_C20121102587686',
'Symbol_C20121110357681',
'Symbol_C20121135',
'Symbol_C20122980',
'Symbol_C20126744',
'Symbol_C20127122',
'Symbol_C20127130',
'Symbol_C20127134',
'Symbol_C20127211',
'Symbol_C20127224',
'Symbol_C20127235',
'Symbol_C20127254',
'Symbol_C20127284',
'Symbol_C20127294',
'Symbol_C20127324',
'Symbol_C20127391',
'Symbol_C20127394',
'Symbol_C20127452',
'Symbol_C20127453',
'Symbol_C20127542',
'Symbol_C20127620',
'Symbol_C20127702',
'Symbol_C20127723',
'Symbol_C20127760',
'Symbol_C20127820',
'Symbol_C20127824',
'Symbol_C20127852',
'Symbol_C20127875',
'Symbol_C20127890',
'Symbol_C20200521',
'Symbol_C20202925',
'Symbol_C20203506',
'Symbol_C20207159',
'Symbol_C20207231',
'Symbol_C20207245',
'Symbol_C20207319',
'Symbol_C20207321',
'Symbol_C20207338',
'Symbol_C20207339',
'Symbol_C20207358',
'Symbol_C20207404',
'Symbol_C20207421',
'Symbol_C20207443',
'Symbol_C20207505',
'Symbol_C20207507',
'Symbol_C20207528',
'Symbol_C20207529',
'Symbol_C20207539',
'Symbol_C20207541',
'Symbol_C20207543',
'Symbol_C20207609',
'Symbol_C20207648',
'Symbol_C20207721',
'Symbol_C20207732',
'Symbol_C20207801',
'Symbol_C20207909',
'Symbol_C20207923',
'Symbol_C20207934',
'Symbol_C20207936',
'Symbol_C20210925',
'Symbol_C20212925',
'Symbol_C20216155',
'Symbol_C20217105',
'Symbol_C20217131',
'Symbol_C20217142',
'Symbol_C20217146',
'Symbol_C20217164',
'Symbol_C20217165',
'Symbol_C20217302',
'Symbol_C20217323',
'Symbol_C20217329',
'Symbol_C20217503',
'Symbol_C20217506',
'Symbol_C20217528',
'Symbol_C20217530',
'Symbol_C20217532',
'Symbol_C20217534',
'Symbol_C20217537',
'Symbol_C20217608',
'Symbol_C20217636',
'Symbol_C20217738',
'Symbol_C20217746',
'Symbol_C20217813',
'Symbol_C20217814',
'Symbol_C20217842',
'Symbol_C20217933',
'Symbol_C20217940',
'Symbol_C20222350',
'Symbol_C20225540',
'Symbol_C20227154',
'Symbol_C20227206',
'Symbol_C20227234',
'Symbol_C20227316',
'Symbol_C20227321',
'Symbol_C20227324',
'Symbol_C20227434',
'Symbol_C20227503',
'Symbol_C20227505',
'Symbol_C20227530',
'Symbol_C20227533',
'Symbol_C20227535',
'Symbol_C20227700',
'Symbol_C20227733',
'Symbol_C20227806',
'Symbol_C20227850',
'Symbol_C20227900',
'Symbol_C20227903',
'Symbol_C20227922',
'Symbol_C20227923',
'Symbol_C20227926',
'Symbol_C21000883',
'Symbol_C21001326',
'Symbol_C21001913',
'Symbol_C21001964',
'Symbol_C21002897',
'Symbol_C21003379',
'Symbol_C21003452',
'Symbol_C21006471',
'Symbol_C21007039',
'Symbol_C21007087',
'Symbol_C21007154',
'Symbol_C21007183',
'Symbol_C21007257',
'Symbol_C21007299',
'Symbol_C21007313',
'Symbol_C21007322',
'Symbol_C21007334',
'Symbol_C21007361',
'Symbol_C21007376',
'Symbol_C21007439',
'Symbol_C21007455',
'Symbol_C21007477',
'Symbol_C21007563',
'Symbol_C21007611',
'Symbol_C21007617',
'Symbol_C21007628',
'Symbol_C21007637',
'Symbol_C21007639',
'Symbol_C21007652',
'Symbol_C21007654',
'Symbol_C21007665',
'Symbol_C21007673',
'Symbol_C21007682',
'Symbol_C21007684',
'Symbol_C21007686',
'Symbol_C21007687',
'Symbol_C21007699',
'Symbol_C21007814',
'Symbol_C21007821',
'Symbol_C21007843',
'Symbol_C21007857',
'Symbol_C21007873',
'Symbol_C21007889',
'Symbol_C21007912',
'Symbol_C21007928',
'Symbol_C21007939',
'Symbol_C21007952',
'Symbol_C21007955',
'Symbol_C21007965',
'Symbol_C21007997',
'Symbol_C21010443',
'Symbol_C21010835',
'Symbol_C21010896',
'Symbol_C21010935',
'Symbol_C21010953',
'Symbol_C21011824',
'Symbol_C21011916',
'Symbol_C21011938',
'Symbol_C21012329',
'Symbol_C21016559',
'Symbol_C21016914',
'Symbol_C21017118',
'Symbol_C21017128',
'Symbol_C21017194',
'Symbol_C21017318',
'Symbol_C21017331',
'Symbol_C21017373',
'Symbol_C21017376',
'Symbol_C21017381',
'Symbol_C21017391',
'Symbol_C21017394',
'Symbol_C21017422',
'Symbol_C21017438',
'Symbol_C21017455',
'Symbol_C21017511',
'Symbol_C21017519',
'Symbol_C21017546',
'Symbol_C21017635',
'Symbol_C21017656',
'Symbol_C21017678',
'Symbol_C21017693',
'Symbol_C21017819',
'Symbol_C21017832',
'Symbol_C21017844',
'Symbol_C21017859',
'Symbol_C21017879',
'Symbol_C21017914',
'Symbol_C21017929',
'Symbol_C21017963',
'Symbol_C21017969',
'Symbol_C21020450',
'Symbol_C21021103677688',
'Symbol_C21021103677689',
'Symbol_C21021910',
'Symbol_C21022815',
'Symbol_C21022910',
'Symbol_C21024815',
'Symbol_C21025833',
'Symbol_C21027020',
'Symbol_C21027155',
'Symbol_C21027310',
'Symbol_C21027340',
'Symbol_C21027350',
'Symbol_C21027464',
'Symbol_C21027615',
'Symbol_C21027674',
'Symbol_C21027880',
'Symbol_C21027934',
'Symbol_C21027975',
'Symbol_C21027983',
'Symbol_C21027994',
'Symbol_C21100244',
'Symbol_C21100284',
'Symbol_C21100832',
'Symbol_C21100921',
'Symbol_C2110115451',
'Symbol_C21101624',
'Symbol_C21103556',
'Symbol_C21104531',
'Symbol_C21107132',
'Symbol_C21107139',
'Symbol_C21107181',
'Symbol_C21107185',
'Symbol_C21107191',
'Symbol_C21107232',
'Symbol_C21107247',
'Symbol_C21107274',
'Symbol_C21107299',
'Symbol_C21107313',
'Symbol_C21107333',
'Symbol_C21107339',
'Symbol_C21107363',
'Symbol_C21107397',
'Symbol_C21107399',
'Symbol_C21107415',
'Symbol_C21107441',
'Symbol_C21107487',
'Symbol_C21107493',
'Symbol_C21107515',
'Symbol_C21107533',
'Symbol_C21107539',
'Symbol_C21107562',
'Symbol_C21107583',
'Symbol_C21107597',
'Symbol_C21107624',
'Symbol_C21107667',
'Symbol_C21107691',
'Symbol_C21107806',
'Symbol_C21107829',
'Symbol_C21107847',
'Symbol_C21107864',
'Symbol_C21107897',
'Symbol_C21107922',
'Symbol_C21107937',
'Symbol_C21107938',
'Symbol_C21107941',
'Symbol_C21107948',
'Symbol_C21107959',
'Symbol_C21107967',
'Symbol_C21107969',
'Symbol_C21107983',
'Symbol_C21107994',
'Symbol_C21110777',
'Symbol_C21110845',
'Symbol_C21110985',
'Symbol_C21111923',
'Symbol_C21112289',
'Symbol_C21112849',
'Symbol_C21112891',
'Symbol_C21112894',
'Symbol_C21117022',
'Symbol_C21117040',
'Symbol_C21117063',
'Symbol_C21117085',
'Symbol_C21117217',
'Symbol_C21117244',
'Symbol_C21117258',
'Symbol_C21117262',
'Symbol_C21117284',
'Symbol_C21117304',
'Symbol_C21117333',
'Symbol_C21117339',
'Symbol_C21117341',
'Symbol_C21117354',
'Symbol_C21117385',
'Symbol_C21117393',
'Symbol_C21117394',
'Symbol_C21117429',
'Symbol_C21117465',
'Symbol_C21117496',
'Symbol_C21117498',
'Symbol_C21117561',
'Symbol_C21117591',
'Symbol_C21117643',
'Symbol_C21117652',
'Symbol_C21117668',
'Symbol_C21117681',
'Symbol_C21117777',
'Symbol_C21117798',
'Symbol_C21117820',
'Symbol_C21117851',
'Symbol_C21117852',
'Symbol_C21117903',
'Symbol_C21117913',
'Symbol_C21120852',
'Symbol_C21120853',
'Symbol_C21121100317683',
'Symbol_C21121112457681',
'Symbol_C21121112457682',
'Symbol_C21124244',
'Symbol_C21127130',
'Symbol_C21127134',
'Symbol_C21127144',
'Symbol_C21127225',
'Symbol_C21127230',
'Symbol_C21127240',
'Symbol_C21127253',
'Symbol_C21127264',
'Symbol_C21127370',
'Symbol_C21127425',
'Symbol_C21127452',
'Symbol_C21127453',
'Symbol_C21127480',
'Symbol_C21127613',
'Symbol_C21127624',
'Symbol_C21127650',
'Symbol_C21127660',
'Symbol_C21127681',
'Symbol_C21127890',
'Symbol_C21127894',
'Symbol_C21127944',
'Symbol_C21127965',
'Symbol_C21127993',
'Symbol_C21200815',
'Symbol_C21200843',
'Symbol_C21201201',
'Symbol_C21202551',
'Symbol_C21207252',
'Symbol_C21207302',
'Symbol_C21207415',
'Symbol_C21207603',
'Symbol_C21207628',
'Symbol_C21207644',
'Symbol_C21207667',
'Symbol_C21207859',
'Symbol_C21207905',
'Symbol_C21207909',
'Symbol_C21207943',
'Symbol_C21207956',
'Symbol_C21210319',
'Symbol_C21211442',
'Symbol_C21217034',
'Symbol_C21217229',
'Symbol_C21217304',
'Symbol_C21217318',
'Symbol_C21217342',
'Symbol_C21217352',
'Symbol_C21217404',
'Symbol_C21217419',
'Symbol_C21217425',
'Symbol_C21217428',
'Symbol_C21217429',
'Symbol_C21217432',
'Symbol_C21217439',
'Symbol_C21217638',
'Symbol_C21217664',
'Symbol_C21217703',
'Symbol_C21217709',
'Symbol_C21217814',
'Symbol_C21217826',
'Symbol_C21217903',
'Symbol_C21217909',
'Symbol_C21217913',
'Symbol_C21217931',
'Symbol_C21217934',
'Symbol_C21217951',
'Symbol_C21217954',
'Symbol_C21222104',
'Symbol_C21222402',
'Symbol_C21222801',
'Symbol_C21222802',
'Symbol_C21222804',
'Symbol_C21223203',
'Symbol_C21227212',
'Symbol_C21227241',
'Symbol_C21227252',
'Symbol_C21227306',
'Symbol_C21227412',
'Symbol_C21227545',
'Symbol_C21227604',
'Symbol_C21227632',
'Symbol_C21227904',
'Symbol_C21227906',
'Symbol_C21227934',
'Symbol_C22000053',
'Symbol_C22000291',
'Symbol_C22007036',
'Symbol_C22007043',
'Symbol_C22007047',
'Symbol_C22007072',
'Symbol_C22007078',
'Symbol_C22007079',
'Symbol_C22007096',
'Symbol_C22007111',
'Symbol_C22007213',
'Symbol_C22007245',
'Symbol_C22007251',
'Symbol_C22007263',
'Symbol_C22007289',
'Symbol_C22007297',
'Symbol_C22007312',
'Symbol_C22007321',
'Symbol_C22007323',
'Symbol_C22007343',
'Symbol_C22007352',
'Symbol_C22007363',
'Symbol_C22007429',
'Symbol_C22007628',
'Symbol_C22007711',
'Symbol_C22007716',
'Symbol_C22007719',
'Symbol_C22007721',
'Symbol_C22007723',
'Symbol_C22007728',
'Symbol_C22007731',
'Symbol_C22007733',
'Symbol_C22007736',
'Symbol_C22007739',
'Symbol_C22007745',
'Symbol_C22007746',
'Symbol_C22007747',
'Symbol_C22007751',
'Symbol_C22007762',
'Symbol_C22007763',
'Symbol_C22007766',
'Symbol_C22007775',
'Symbol_C22007777',
'Symbol_C22007781',
'Symbol_C22007784',
'Symbol_C22007786',
'Symbol_C22010051',
'Symbol_C22010159',
'Symbol_C22011088',
'Symbol_C22017016',
'Symbol_C22017035',
'Symbol_C22017052',
'Symbol_C22017054',
'Symbol_C22017056',
'Symbol_C22017085',
'Symbol_C22017089',
'Symbol_C22017096',
'Symbol_C22017112',
'Symbol_C22017148',
'Symbol_C22017189',
'Symbol_C22017258',
'Symbol_C22017271',
'Symbol_C22017329',
'Symbol_C22017334',
'Symbol_C22017348',
'Symbol_C22017354',
'Symbol_C22017363',
'Symbol_C22017452',
'Symbol_C22017654',
'Symbol_C22017716',
'Symbol_C22017721',
'Symbol_C22017722',
'Symbol_C22017728',
'Symbol_C22017738',
'Symbol_C22017743',
'Symbol_C22017752',
'Symbol_C22017762',
'Symbol_C22017763',
'Symbol_C22017764',
'Symbol_C22017779',
'Symbol_C22017786',
'Symbol_C22020070',
'Symbol_C22027034',
'Symbol_C22027070',
'Symbol_C22027251',
'Symbol_C22027292',
'Symbol_C22027320',
'Symbol_C22027354',
'Symbol_C22027360',
'Symbol_C22027393',
'Symbol_C22027394',
'Symbol_C22027720',
'Symbol_C22027723',
'Symbol_C22027724',
'Symbol_C22027725',
'Symbol_C22027733',
'Symbol_C22027734',
'Symbol_C22027745',
'Symbol_C22027753',
'Symbol_C22027770',
'Symbol_C22027772',
'Symbol_C22027780',
'Symbol_C22027790',
'Symbol_C22100068',
'Symbol_C22100488',
'Symbol_C22101034',
'Symbol_C22101099',
'Symbol_C22102539',
'Symbol_C22105133',
'Symbol_C22107068',
'Symbol_C22107148',
'Symbol_C22107189',
'Symbol_C22107191',
'Symbol_C22107192',
'Symbol_C22107254',
'Symbol_C22107332',
'Symbol_C22107343',
'Symbol_C22107363',
'Symbol_C22107425',
'Symbol_C22107426',
'Symbol_C22107705',
'Symbol_C22107722',
'Symbol_C22107731',
'Symbol_C22107732',
'Symbol_C22107734',
'Symbol_C22107745',
'Symbol_C22107765',
'Symbol_C22107766',
'Symbol_C22107767',
'Symbol_C22107768',
'Symbol_C22107769',
'Symbol_C22107786',
'Symbol_C22107788',
'Symbol_C22107791',
'Symbol_C22107793',
'Symbol_C22107796',
'Symbol_C22110055',
'Symbol_C22110191',
'Symbol_C22110325',
'Symbol_C22113544',
'Symbol_C22117018',
'Symbol_C22117036',
'Symbol_C22117037',
'Symbol_C22117084',
'Symbol_C22117093',
'Symbol_C22117094',
'Symbol_C22117095',
'Symbol_C22117099',
'Symbol_C22117146',
'Symbol_C22117169',
'Symbol_C22117173',
'Symbol_C22117225',
'Symbol_C22117243',
'Symbol_C22117245',
'Symbol_C22117247',
'Symbol_C22117255',
'Symbol_C22117286',
'Symbol_C22117289',
'Symbol_C22117339',
'Symbol_C22117349',
'Symbol_C22117363',
'Symbol_C22117393',
'Symbol_C22117395',
'Symbol_C22117426',
'Symbol_C22117453',
'Symbol_C22117456',
'Symbol_C22117465',
'Symbol_C22117714',
'Symbol_C22117715',
'Symbol_C22117718',
'Symbol_C22117722',
'Symbol_C22117723',
'Symbol_C22117735',
'Symbol_C22117739',
'Symbol_C22117742',
'Symbol_C22117743',
'Symbol_C22117754',
'Symbol_C22117763',
'Symbol_C22117782',
'Symbol_C22117786',
'Symbol_C22117795',
'Symbol_C22117796',
'Symbol_C22117797',
'Symbol_C22120260',
'Symbol_C22127004',
'Symbol_C22127064',
'Symbol_C22127081',
'Symbol_C22127082',
'Symbol_C22127233',
'Symbol_C22127284',
'Symbol_C22127330',
'Symbol_C22127353',
'Symbol_C22127364',
'Symbol_C22127421',
'Symbol_C22127422',
'Symbol_C22127713',
'Symbol_C22127731',
'Symbol_C22127764',
'Symbol_C22127790',
'Symbol_C22127791',
'Symbol_C22207003',
'Symbol_C22207162',
'Symbol_C22207217',
'Symbol_C22207325',
'Symbol_C22207334',
'Symbol_C22207711',
'Symbol_C22207715',
'Symbol_C22207727',
'Symbol_C22207732',
'Symbol_C22207771',
'Symbol_C22207773',
'Symbol_C22207774',
'Symbol_C22207776',
'Symbol_C22207777',
'Symbol_C22207778',
'Symbol_C22210035',
'Symbol_C22210308',
'Symbol_C22212219',
'Symbol_C22217016',
'Symbol_C22217044',
'Symbol_C22217052',
'Symbol_C22217202',
'Symbol_C22217208',
'Symbol_C22217235',
'Symbol_C22217238',
'Symbol_C22217452',
'Symbol_C22217513',
'Symbol_C22217609',
'Symbol_C22217701',
'Symbol_C22217714',
'Symbol_C22217731',
'Symbol_C22217771',
'Symbol_C22217773',
'Symbol_C22217775',
'Symbol_C22217776',
'Symbol_C22217778',
'Symbol_C22217779',
'Symbol_C22220205',
'Symbol_C22221245',
'Symbol_C22222020',
'Symbol_C22227034',
'Symbol_C22227104',
'Symbol_C22227202',
'Symbol_C22227303',
'Symbol_C22227333',
'Symbol_C22227555',
'Symbol_C22227702',
'Symbol_C22227720',
'Symbol_C22227726',
'Symbol_C22227731',
'Symbol_C22227735',
'Symbol_C22227770',
'Symbol_C22227772',
'Symbol_C22227773',
'Symbol_C22227774',
'Symbol_C22227776',
'VenueType_Dark',
'Side_Buy',
'Side_Sell',
'SecurityCategory_ADR',
'SecurityCategory_COMMON',
'SecurityCategory_ETF',
'SecurityCategory_FUND',
'SecurityCategory_MISC',
'SecurityCategory_OTHER_DEP_RCPT',
'SecurityCategory_PREFERRED',
'SecurityCategory_REIT',
'SecurityCategory_UNIT',
'SecurityCategory_WARRANT',
'Sector_Basic Materials',
'Sector_Communications',
'Sector_Consumer Cyclical',
'Sector_Consumer Non-cyclical',
'Sector_Diversified',
'Sector_Energy',
'Sector_Financial',
'Sector_Government',
'Sector_Industrial',
'Sector_Technology',
'Sector_Utilities',
'MktCap_LARGE',
'MktCap_MID',
'MktCap_SMALL'
]

from sklearn.preprocessing import PolynomialFeatures


poly = PolynomialFeatures(degree=3, include_bias=False)

poly.fit(combined_training[polyFeaturesBase])

TrainingFeatures_list = poly.transform(combined_training[polyFeaturesBase])
TestingFeatures_list  = poly.transform(combined_testing[polyFeaturesBase])

TrainingFeatures = pd.DataFrame(TrainingFeatures_list, columns=poly.get_feature_names())
TestingFeatures  = pd.DataFrame(TestingFeatures_list,  columns=poly.get_feature_names())

polyFeatureNames = poly.get_feature_names()

for i in range(len(polyFeaturesBase)):
    polyFeatureNames.remove(polyFeatureNames[0])
    TrainingFeatures = TrainingFeatures.drop(poly.get_feature_names()[i], axis=1)
    TestingFeatures  = TestingFeatures.drop( poly.get_feature_names()[i], axis=1)
    
TrainingFeatures['OrderID'] = list(combined_training.index)
TestingFeatures[ 'OrderID'] = list(combined_testing.index)

TrainingFeatures.set_index('OrderID', inplace= True)
TestingFeatures.set_index( 'OrderID', inplace= True)

combined_training = pd.concat([combined_training, TrainingFeatures], axis=1)
combined_testing  = pd.concat([combined_testing,  TestingFeatures],  axis=1)

cols = [col for col in combined_testing.columns if col not in ['BecomeTrade','StartTime','TradeDateTime','ContinuousStartTime']]
X_test = combined_testing[cols]
X_test.fillna(0, inplace = True)
y_test = combined_testing['BecomeTrade']

cols1 = [col for col in combined_training.columns if col not in ['BecomeTrade','TradeDateTime']]
X_train = combined_training[cols1]
X_train.fillna(0, inplace = True)
y_train = combined_training['BecomeTrade']

from sklearn import preprocessing
#from sklearn_pandas import DataFrameMapper

for name in normalizationNames:
    scaler = preprocessing.StandardScaler()
    X_train.loc[:,name] = scaler.fit_transform(X_train[name].values.reshape(-1,1))
    X_test.loc[:,name]  = scaler.transform(X_test[name].values.reshape(-1,1))
    
for name in polyFeatureNames:
    scaler = preprocessing.StandardScaler()
    X_train.loc[:,name] = scaler.fit_transform(X_train[name].values.reshape(-1,1))
    X_test.loc[:,name]  = scaler.transform(X_test[name].values.reshape(-1,1))
    
from sklearn.decomposition import PCA as sklearnPCA
# Preprocessing Data for PCA
X_train_1 = X_train[normalizationNames + polyFeatureNames]
X_train_1.fillna(0, inplace = True)

X_test_1 = X_test[normalizationNames + polyFeatureNames]
X_test_1.fillna(0, inplace = True)

# PCA
sklearn_pca = sklearnPCA(.95)
sklearn_pca.fit(X_train_1)

# Plot PCA (for internal analysis)
import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.plot(np.cumsum(sklearn_pca.explained_variance_ratio_))

plt.figure(2)
plt.xlabel('number of components')
plt.ylabel('explained variance')
plt.plot(sklearn_pca.explained_variance_ratio_)

X_train_pca = sklearn_pca.transform(X_train_1)
X_test_pca  = sklearn_pca.transform(X_test_1)


cols_pca=[
'PCA_0',
'PCA_1',
'PCA_2',
'PCA_3',
'PCA_4',
'PCA_5',
'PCA_6',
'PCA_7',
'PCA_8',
'PCA_9',
'PCA_10',
'PCA_11',
'PCA_12',
'PCA_13',
'PCA_14',
'PCA_15',
'PCA_16',
'PCA_17',
'PCA_18',
'PCA_19',
'PCA_20',
'PCA_21',
'PCA_22',
'PCA_23',
'PCA_24',
'PCA_25',
'PCA_26',
'PCA_27',
'PCA_28',
'PCA_29',
'PCA_30'
]

PCA_train = pd.DataFrame(data = X_train_pca, columns = cols_pca)
PCA_test = pd.DataFrame(data = X_test_pca, columns = cols_pca)

PCA_train['OrderID'] = list(combined_training.index)
PCA_test[ 'OrderID'] = list(combined_testing.index)

PCA_train.set_index('OrderID', inplace= True)
PCA_test.set_index( 'OrderID', inplace= True)


X_train = pd.concat((PCA_train, combined_training[unormalizedNames]), axis=1)
X_test  = pd.concat((PCA_test,  combined_testing[unormalizedNames]),  axis=1)

X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)

time_x_train = X_train[['StartTime','ContinuousStartTime']]

cols2 = [col for col in X_test.columns if col not in ['BecomeTrade','StartTime','TradeDateTime','ContinuousStartTime']]
TEMP_X_test = X_test[cols2]

cols3 = [col for col in X_train.columns if col not in ['BecomeTrade','TradeDateTime','StartTime','ContinuousStartTime']]
TEMP_X_train = X_train[cols3]

"""Feature Selection"""

from sklearn.ensemble import ExtraTreesClassifier
rf = ExtraTreesClassifier()
rf.fit(TEMP_X_train, y_train)
feature_importance = list(rf.feature_importances_)


from sklearn.feature_selection import SelectFromModel

model=SelectFromModel(rf,threshold="mean", prefit=True)


X_new=model.transform(TEMP_X_train)

feature_idx = model.get_support()
feature_name = TEMP_X_train.columns[feature_idx]


train_temp_x = TEMP_X_train[feature_name]
test_temp_x = TEMP_X_test[feature_name]

#X_train, X_test are for stacked ensemble method"""

X_train = pd.concat([train_temp_x,time_x_train],axis=1)
X_test = test_temp_x



# raw data on LR

clf = LogisticRegression()
clf.fit(train_temp_x,y_train)
y_score = clf.decision_function(test_temp_x)
precision, recall, _ = precision_recall_curve(y_test, y_score)

pre_train_LR_Raw = clf.predict(train_temp_x)
recall_train_LR_Raw = recall_score(y_train,pre_train_LR_Raw)
precision_train_LR_Raw = precision_score(y_train,pre_train_LR_Raw)
acc_train_LR_Raw = clf.score(train_temp_x, y_train)
f1_train_LR_Raw = f1_score(y_train,pre_train_LR_Raw)

pre_test_LR_Raw = clf.predict(test_temp_x)
clf_proba = clf.predict_proba(test_temp_x)[:,1]
clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, clf_proba)
clf_auc = auc(clf_fpr, clf_tpr)

recall_test_LR_Raw = recall_score(y_test, pre_test_LR_Raw)
precision_test_LR_Raw = precision_score(y_test, pre_test_LR_Raw)
acc_test_LR_Raw = clf.score(test_temp_x, y_test)
f1_test_LR_Raw = f1_score(y_test, pre_test_LR_Raw)


# raw data on RandomForest

clf = RandomForestClassifier(n_estimators=100,max_depth = 15)
clf.fit(train_temp_x,y_train)
y_score = clf.predict_proba(test_temp_x)[:,1]
precision1, recall1, _ = precision_recall_curve(y_test, y_score)

pre_train_RF_Raw = clf.predict(train_temp_x)
recall_train_RF_Raw = recall_score(y_train,pre_train_RF_Raw)
precision_train_RF_Raw = precision_score(y_train,pre_train_RF_Raw)
acc_train_RF_Raw = clf.score(train_temp_x, y_train)
f1_train_RF_Raw = f1_score(y_train,pre_train_RF_Raw)

pre_test_RF_Raw = clf.predict(test_temp_x)
clf_proba1 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr1, clf_tpr1, clf_thresholds1 = roc_curve(y_test, clf_proba1)
clf_auc1 = auc(clf_fpr1, clf_tpr1)

recall_test_RF_Raw = recall_score(y_test, pre_test_RF_Raw)
precision_test_RF_Raw = precision_score(y_test, pre_test_RF_Raw)
acc_test_RF_Raw = clf.score(test_temp_x, y_test)
f1_test_RF_Raw = f1_score(y_test, pre_test_RF_Raw)


# raw data on Adaboosting

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(train_temp_x,y_train)
y_score = clf.decision_function(test_temp_x)
precision2, recall2, _ = precision_recall_curve(y_test, y_score)

pre_train_AB_Raw = clf.predict(train_temp_x)
recall_train_AB_Raw = recall_score(y_train,pre_train_AB_Raw)
precision_train_AB_Raw = precision_score(y_train,pre_train_AB_Raw)
acc_train_AB_Raw = clf.score(train_temp_x, y_train)
f1_train_AB_Raw = f1_score(y_train,pre_train_AB_Raw)



pre_test_AB_Raw = clf.predict(test_temp_x)
clf_proba2 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr2, clf_tpr2, clf_thresholds2 = roc_curve(y_test, clf_proba2)
clf_auc2 = auc(clf_fpr2, clf_tpr2)

recall_test_AB_Raw = recall_score(y_test, pre_test_AB_Raw)
precision_test_AB_Raw = precision_score(y_test, pre_test_AB_Raw)
acc_test_AB_Raw = clf.score(test_temp_x, y_test)
f1_test_AB_Raw = f1_score(y_test, pre_test_AB_Raw)

plt.figure()

#plt.plot([0,1],[0,1], 'k--')
plt.plot(precision,recall, label='Raw Data + LR')
plt.plot(precision1,recall1, label='Raw Data + RF')
plt.plot(precision2,recall2,label='Raw Data + AB')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.ylim([0.0,1.05])
plt.xlim([0.15,0.5])
plt.legend(loc='best')
plt.show()



plt.figure()

plt.plot([0,1],[0,1], 'k--')
plt.plot(clf_fpr,clf_tpr, label='Raw Data + LR')
plt.plot(clf_fpr1,clf_tpr1, label='Raw Data + RF')
plt.plot(clf_fpr2,clf_tpr2, label='Raw Data + AB')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# raw data with resampling + AdaBoost + testing resampling rate

df_f=pd.concat([train_temp_x,y_train],axis=1)

training_0 = df_f[df_f['BecomeTrade'] == 0]
training_1 = df_f[df_f['BecomeTrade'] == 1]

resamplePercent = 0.1

Precision_for_pca_train = {}
recall_for_pca_train = {}
accuracy_for_pca_train = {}
f1_for_pca_train = {}

Precision_for_pca_test = {}
recall_for_pca_test = {}
accuracy_for_pca_test = {}
f1_for_pca_test = {}

while (resamplePercent <= 0.9):
    alpha = np.ceil((float(len(training_0))/float(len(training_1)))*resamplePercent)
    
    combined_training = pd.concat([training_1]*int(alpha), ignore_index = True).append(training_0)
    
    X2_1 = combined_training.loc[:,combined_training.columns!='BecomeTrade']
    df_train_y_1 = combined_training['BecomeTrade']
    
    #clf = AdaBoostClassifier(n_estimators=100)
    clf = LogisticRegression()
    clf.fit(X2_1,df_train_y_1)
    
    clf_predict_train = clf.predict(X2_1)
    clf_recall_train = recall_score(df_train_y_1,clf_predict_train)
    clf_precision_train = precision_score(df_train_y_1,clf_predict_train)
    clf_acc_train = clf.score(X2_1, df_train_y_1)
    clf_f1_train = f1_score(df_train_y_1,clf_predict_train)


    clf_predict = clf.predict(test_temp_x)
    
    clf_recall = recall_score(y_test, clf_predict)
    clf_precision = precision_score(y_test, clf_predict)
    clf_acc = clf.score(test_temp_x, y_test)
    clf_f1 = f1_score(y_test, clf_predict)
    
    print ('Accuracy: %f \n' % clf.score(test_temp_x, y_test))
    #print ("AUC: %f \n" % clf_auc)
    print ("Recall Score: %f \n" % clf_recall)
    print ("Precision Score: %f \n" % clf_precision)
    print ("Confusion matrix:")
    print (pd.crosstab(y_test, clf_predict))
    
    Precision_for_pca_train[resamplePercent] = clf_precision_train
    recall_for_pca_train[resamplePercent] = clf_recall_train
    accuracy_for_pca_train[resamplePercent] = clf_acc_train
    f1_for_pca_train[resamplePercent] = clf_f1_train
    
    Precision_for_pca_test[resamplePercent] = clf_precision
    recall_for_pca_test[resamplePercent] = clf_recall
    accuracy_for_pca_test[resamplePercent] = clf_acc
    f1_for_pca_test[resamplePercent] = clf_f1
    
    resamplePercent += 0.1

list_f1_test_LR = sorted(f1_for_pca_test.items())
x1,y1 = zip(*list_f1_test_LR)

list_f1_test_AB = sorted(f1_for_pca_test.items())
x2,y2 = zip(*list_f1_test_AB)

list_recall_test_LR = sorted(recall_for_pca_test.items())
x3,y3 = zip(*list_recall_test_LR)

list_recall_test_AB = sorted(recall_for_pca_test.items())
x4,y4 = zip(*list_recall_test_AB)

list_Precision_test_LR = sorted(Precision_for_pca_test.items())
x5,y5 = zip(*list_Precision_test_LR)

list_precision_test_AB = sorted(Precision_for_pca_test.items())
x6,y6 = zip(*list_precision_test_AB)


plt.figure(figsize=(8,10))

#plt.plot([0,1],[0,1], 'k--')
plt.plot(x1,y1,label='LogisticRegression')
plt.plot(x2,y2,label='AdaBoost')

plt.xlabel('Oversampling Rate')
plt.ylabel('f1 score')
#plt.ylim([0.0,1.05])
#plt.xlim([0.18,0.4])
plt.legend(loc='upper left')
plt.show()


# raw data with resampling and AdaBoost
df_f=pd.concat([train_temp_x,y_train],axis=1)

training_0 = df_f[df_f['BecomeTrade'] == 0]
training_1 = df_f[df_f['BecomeTrade'] == 1]
resamplePercent = 0.45

alpha = np.ceil((float(len(training_0))/float(len(training_1)))*resamplePercent)
combined_training = pd.concat([training_1]*int(alpha), ignore_index = True).append(training_0)
    
X2_1 = combined_training.loc[:,combined_training.columns!='BecomeTrade']
df_train_y_1 = combined_training['BecomeTrade']
    
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X2_1,df_train_y_1)
y_score = clf.decision_function(test_temp_x)
precision3, recall3, _ = precision_recall_curve(y_test, y_score)
    
predict_train_AB_Resample = clf.predict(X2_1)
recall_train_AB_Resample = recall_score(df_train_y_1,predict_train_AB_Resample)
precision_train_AB_Resample = precision_score(df_train_y_1,predict_train_AB_Resample)
acc_train_AB_Resample = clf.score(X2_1, df_train_y_1)
f1_train_AB_Resample = f1_score(df_train_y_1,predict_train_AB_Resample)


predict_test_AB_Resample = clf.predict(test_temp_x)
clf_proba3 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr3, clf_tpr3, clf_thresholds3 = roc_curve(y_test, clf_proba3)
clf_auc3 = auc(clf_fpr3, clf_tpr3)
    
recall_test_AB_Resample = recall_score(y_test, predict_test_AB_Resample)
precision_test_AB_Resample = precision_score(y_test, predict_test_AB_Resample)
acc_test_AB_Resample = clf.score(test_temp_x, y_test)
f1_test_AB_Resample = f1_score(y_test, predict_test_AB_Resample)


# undersampling + AdaBoost

data_25 = pd.read_csv('xxx/df_25.csv')
data_50 = pd.read_csv('xxx/df_50.csv')
data_75 = pd.read_csv('xxx/df_75.csv')


data_25=data_25.drop(['Unnamed: 0'],axis=1)

data_25.set_index('OrderID',inplace=True)
data_50.set_index('OrderID',inplace=True)
data_75.set_index('OrderID',inplace=True)


data_x_25 = data_25.loc[:,data_25.columns!='BecomeTrade']
data_x_25=data_x_25[feature_name]
data_y_25 =  data_25['BecomeTrade']

data_x_50 = data_50.loc[:,data_50.columns!='BecomeTrade']
data_x_50=data_x_50[feature_name]
data_y_50 =  data_50['BecomeTrade']

data_x_75 = data_75.loc[:,data_75.columns!='BecomeTrade']
data_x_75=data_x_75[feature_name]
data_y_75 =  data_75['BecomeTrade']



clf = AdaBoostClassifier(n_estimators=100)
clf.fit(data_x_75,data_y_75)
y_score = clf.decision_function(test_temp_x)
precision8, recall8, _ = precision_recall_curve(y_test, y_score)

clf_predict_train_75 = clf.predict(data_x_75)
clf_recall_train_75 = recall_score(data_y_75,clf_predict_train_75)
clf_precision_train_75 = precision_score(data_y_75,clf_predict_train_75)
clf_acc_train_75 = clf.score(data_x_75, data_y_75)
clf_f1_train_75 = f1_score(data_y_75,clf_predict_train_75)

clf_proba4 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr4, clf_tpr4, clf_thresholds4 = roc_curve(y_test, clf_proba4)
clf_auc4 = auc(clf_fpr4, clf_tpr4)
    

clf_predict_75 = clf.predict(test_temp_x)
clf_recall_75 = recall_score(y_test, clf_predict_75)
clf_precision_75 = precision_score(y_test, clf_predict_75)
clf_acc_75 = clf.score(test_temp_x, y_test)
clf_f1_75 = f1_score(y_test, clf_predict_75)


clf = AdaBoostClassifier(n_estimators=100)
clf.fit(data_x_25,data_y_25)
y_score = clf.decision_function(test_temp_x)
precision7, recall7, _ = precision_recall_curve(y_test, y_score)

clf_predict_train_25 = clf.predict(data_x_25)
clf_recall_train_25 = recall_score(data_y_25,clf_predict_train_25)
clf_precision_train_25 = precision_score(data_y_25,clf_predict_train_25)
clf_acc_train_25 = clf.score(data_x_25, data_y_25)
clf_f1_train_25 = f1_score(data_y_25,clf_predict_train_25)

clf_proba5 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr5, clf_tpr5, clf_thresholds5 = roc_curve(y_test, clf_proba5)
clf_auc5 = auc(clf_fpr5, clf_tpr5)
    

clf_predict_25 = clf.predict(test_temp_x)
clf_recall_25 = recall_score(y_test, clf_predict_25)
clf_precision_25 = precision_score(y_test, clf_predict_25)
clf_acc_25 = clf.score(test_temp_x, y_test)
clf_f1_25 = f1_score(y_test, clf_predict_25)

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(data_x_50,data_y_50)

y_score = clf.decision_function(test_temp_x)
precision4, recall4, _ = precision_recall_curve(y_test, y_score)

clf_predict_train_50 = clf.predict(data_x_50)
clf_recall_train_50 = recall_score(data_y_50,clf_predict_train_50)
clf_precision_train_50 = precision_score(data_y_50,clf_predict_train_50)
clf_acc_train_50 = clf.score(data_x_50, data_y_50)
clf_f1_train_50 = f1_score(data_y_50,clf_predict_train_50)

clf_proba6 = clf.predict_proba(test_temp_x)[:,1]
clf_fpr6, clf_tpr6, clf_thresholds6 = roc_curve(y_test, clf_proba6)
clf_auc6 = auc(clf_fpr6, clf_tpr6)
    

clf_predict_50 = clf.predict(test_temp_x)
clf_recall_50 = recall_score(y_test, clf_predict_50)
clf_precision_50 = precision_score(y_test, clf_predict_50)
clf_acc_50 = clf.score(test_temp_x, y_test)
clf_f1_50 = f1_score(y_test, clf_predict_50)



plt.figure()

#plt.plot([0,1],[0,1], 'k--')
plt.plot(precision8,recall8, label='Undersampling 75 + AB')
plt.plot(precision7,recall7, label='Undersampling 25 + AB')
plt.plot(precision4,recall4, label='Undersampling 50 + AB')

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precison Recall Curve')
plt.xlim([0.18,0.45])
plt.ylim([0.0,1.0])
plt.legend(loc='best')
plt.show()

# Random Sampling Code

Train = pd.concat((X_train, y_train),axis=1)



Train['date']=Train['StartTime'].dt.date

Train.drop(['StartTime'],axis=1,inplace=True)

Train_1 = Train[Train['BecomeTrade'] == 1]
Train_0 = Train[Train['BecomeTrade'] == 0]
M = len(Train_1)
N = len(Train_0)
Train_0 = Train_0.sort_values(by=['date','ContinuousStartTime'])
Train_0['counter']=range(N)

Train_1 = Train_1.drop(['date','ContinuousStartTime'],1)
Train_0 = Train_0.drop(['date','ContinuousStartTime'],1)


cols4 = [col for col in Train.columns if col not in ['BecomeTrade','StartTime','TradeDateTime','ContinuousStartTime','date']]
x_train = Train[cols4]
y_train = Train['BecomeTrade']



SS = [
'Prediction_0',
'Prediction_1',
'Prediction_2',
'Prediction_3',
'Prediction_4',
'Prediction_5',
'Prediction_6',
'Prediction_7',
'Prediction_8',
'Prediction_9',
'Prediction_10',
'Prediction_11',
'Prediction_12',
'Prediction_13',
'Prediction_14',
'Prediction_15',
'Prediction_16',
'Prediction_17',
'Prediction_18',
'Prediction_19',
'Prediction_20',
'Prediction_21',
'Prediction_22',
'Prediction_23',
'Prediction_24',
'Prediction_25',
'Prediction_26',
'Prediction_27',
'Prediction_28',
'Prediction_29',
'Prediction_30',
'Prediction_31',
'Prediction_32',
'Prediction_33',
'Prediction_34',
'Prediction_35',
'Prediction_36',
'Prediction_37',
'Prediction_38',
'Prediction_39',
'Prediction_40',
'Prediction_41',
'Prediction_42',
'Prediction_43',
'Prediction_44',
'Prediction_45',
'Prediction_46',
'Prediction_47',
'Prediction_48',
'Prediction_49',
'Prediction_50',
'Prediction_51',
'Prediction_52',
'Prediction_53',
'Prediction_54',
'Prediction_55',
'Prediction_56',
'Prediction_57',
'Prediction_58',
'Prediction_59',
'Prediction_60',
'Prediction_61',
'Prediction_62',
'Prediction_63',
'Prediction_64',
'Prediction_65',
'Prediction_66',
'Prediction_67',
'Prediction_68',
'Prediction_69',
'Prediction_70',
'Prediction_71',
'Prediction_72',
'Prediction_73',
'Prediction_74',
'Prediction_75',
'Prediction_76',
'Prediction_77',
'Prediction_78',
'Prediction_79',
'Prediction_80',
'Prediction_81',
'Prediction_82',
'Prediction_83',
'Prediction_84',
'Prediction_85',
'Prediction_86',
'Prediction_87',
'Prediction_88',
'Prediction_89',
'Prediction_90',
'Prediction_91',
'Prediction_92',
'Prediction_93',
'Prediction_94',
'Prediction_95',
'Prediction_96',
'Prediction_97',
'Prediction_98',
'Prediction_99',
'Prediction_100',
'Prediction_101',
'Prediction_102',
'Prediction_103',
'Prediction_104',
'Prediction_105',
'Prediction_106',
'Prediction_107',
'Prediction_108',
'Prediction_109',
'Prediction_110',
'Prediction_111',
'Prediction_112',
'Prediction_113',
'Prediction_114',
'Prediction_115',
'Prediction_116',
'Prediction_117',
'Prediction_118',
'Prediction_119',
'Prediction_120',
'Prediction_121',
'Prediction_122',
'Prediction_123',
'Prediction_124',
'Prediction_125',
'Prediction_126',
'Prediction_127',
'Prediction_128',
'Prediction_129',
'Prediction_130',
'Prediction_131',
'Prediction_132',
'Prediction_133',
'Prediction_134',
'Prediction_135',
'Prediction_136',
'Prediction_137',
'Prediction_138',
'Prediction_139',
'Prediction_140',
'Prediction_141',
'Prediction_142',
'Prediction_143',
'Prediction_144',
'Prediction_145',
'Prediction_146',
'Prediction_147',
'Prediction_148',
'Prediction_149',
'Prediction_150',
'Prediction_151',
'Prediction_152',
'Prediction_153',
'Prediction_154',
'Prediction_155',
'Prediction_156',
'Prediction_157',
'Prediction_158',
'Prediction_159',
'Prediction_160',
'Prediction_161',
'Prediction_162',
'Prediction_163',
'Prediction_164',
'Prediction_165',
'Prediction_166',
'Prediction_167',
'Prediction_168',
'Prediction_169',
'Prediction_170',
'Prediction_171',
'Prediction_172',
'Prediction_173',
'Prediction_174',
'Prediction_175',
'Prediction_176',
'Prediction_177',
'Prediction_178',
'Prediction_179',
'Prediction_180',
'Prediction_181',
'Prediction_182',
'Prediction_183',
'Prediction_184',
'Prediction_185',
'Prediction_186',
'Prediction_187',
'Prediction_188',
'Prediction_189',
'Prediction_190',
'Prediction_191',
'Prediction_192',
'Prediction_193',
'Prediction_194',
'Prediction_195',
'Prediction_196',
'Prediction_197',
'Prediction_198',
'Prediction_199'
]

import random
from sklearn.neural_network import MLPClassifier

# generate random beginning number and select bootstrap samples

S1=pd.DataFrame(data=None,columns=SS, index=x_train.index)
S2=pd.DataFrame(data=None,columns=SS, index=X_test.index)
i=0
while (i < 200):
    n = random.randint(0,N)
    if n <= (N-M):
        list0 = list(range(n, n+M))
        temp_0 = Train_0.loc[Train_0['counter'].isin(list0)]
        temp_0 = temp_0.drop('counter', 1)
        temp = pd.concat((temp_0,Train_1),axis=0)
        temp_x = temp.loc[:, temp.columns != 'BecomeTrade']
        temp_y = temp['BecomeTrade']
        clf = LogisticRegression()
        clf.fit(temp_x, temp_y)
        clf_predict_train = clf.predict(x_train)
        clf_predict_test = clf.predict(X_test)
        name = SS[i]
        S1[name] = clf_predict_train
        S2[name] = clf_predict_test
        del clf_predict_train
        del clf_predict_test
        del temp_0
        del temp_x
        del temp_y
        del temp
        i = i + 1
    else:
        continue
    
Train_f_x = pd.concat((x_train, S1), axis = 1)
Test_f_x = pd.concat((X_test, S2), axis = 1)

df_f=pd.concat([Train_f_x,y_train],axis=1)

training_0 = df_f[df_f['BecomeTrade'] == 0]
training_1 = df_f[df_f['BecomeTrade'] == 1]


resamplePercent = 0.1

Precision_for_pca_train = {}
recall_for_pca_train = {}
accuracy_for_pca_train = {}
f1_for_pca_train = {}

Precision_for_pca_test = {}
recall_for_pca_test = {}
accuracy_for_pca_test = {}
f1_for_pca_test = {}

from sklearn.linear_model import LogisticRegression

while (resamplePercent <= 0.9):
    alpha = np.ceil((float(len(training_0))/float(len(training_1)))*resamplePercent)
    
    combined_training = pd.concat([training_1]*int(alpha), ignore_index = True).append(training_0)
    
    X2_1 = combined_training.loc[:,combined_training.columns!='BecomeTrade']
    df_train_y_1 = combined_training['BecomeTrade']

    clf = RandomForestClassifier(n_estimators=100,random_state=0)
    #clf = MLPClassifier(hidden_layer_sizes=(25,10),activation='relu')
    #clf = LogisticRegression()

    clf.fit(X2_1,df_train_y_1)
    
    clf_predict_train = clf.predict(X2_1)
    clf_recall_train = recall_score(df_train_y_1,clf_predict_train)
    clf_precision_train = precision_score(df_train_y_1,clf_predict_train)
    clf_acc_train = clf.score(X2_1, df_train_y_1)
    clf_f1_train = f1_score(df_train_y_1,clf_predict_train)

    clf_predict = clf.predict(Test_f_x)

    clf_recall = recall_score(y_test, clf_predict)
    clf_precision = precision_score(y_test, clf_predict)
    clf_acc = clf.score(Test_f_x, y_test)
    clf_f1 = f1_score(y_test, clf_predict)
    
    print ('Accuracy: %f \n' % clf.score(Test_f_x, y_test))
    #print ("AUC: %f \n" % clf_auc)
    print ("Recall Score: %f \n" % clf_recall)
    print ("Precision Score: %f \n" % clf_precision)
    print ("Confusion matrix:")
    print (pd.crosstab(y_test, clf_predict))
    
    Precision_for_pca_train[resamplePercent] = clf_precision_train
    recall_for_pca_train[resamplePercent] = clf_recall_train
    accuracy_for_pca_train[resamplePercent] = clf_acc_train
    f1_for_pca_train[resamplePercent] = clf_f1_train
    
    Precision_for_pca_test[resamplePercent] = clf_precision
    recall_for_pca_test[resamplePercent] = clf_recall
    accuracy_for_pca_test[resamplePercent] = clf_acc
    f1_for_pca_test[resamplePercent] = clf_f1
    
    
    resamplePercent += 0.1


# Split sequentially equal size data

SS = [
'Prediction_0',
'Prediction_1',
'Prediction_2',
'Prediction_3',
'Prediction_4',
'Prediction_5',
'Prediction_6',
'Prediction_7',
'Prediction_8',
'Prediction_9',
'Prediction_10',
'Prediction_11',
'Prediction_12',
'Prediction_13',
'Prediction_14',
'Prediction_15',
'Prediction_16',
'Prediction_17',
'Prediction_18',
'Prediction_19',
'Prediction_20',
'Prediction_21',
'Prediction_22',
'Prediction_23',
'Prediction_24',
'Prediction_25',
'Prediction_26',
'Prediction_27',
'Prediction_28',
'Prediction_29',
'Prediction_30',
'Prediction_31',
'Prediction_32',
'Prediction_33',
'Prediction_34',
'Prediction_35',
'Prediction_36',
'Prediction_37',
'Prediction_38',
'Prediction_39',
'Prediction_40',
'Prediction_41',
'Prediction_42',
'Prediction_43',
'Prediction_44',
'Prediction_45',
'Prediction_46',
'Prediction_47',
'Prediction_48',
'Prediction_49',
'Prediction_50',
'Prediction_51',
'Prediction_52',
'Prediction_53',
'Prediction_54',
'Prediction_55',
'Prediction_56',
'Prediction_57',
'Prediction_58',
'Prediction_59',
'Prediction_60',
'Prediction_61',
'Prediction_62',
'Prediction_63',
'Prediction_64',
'Prediction_65',
'Prediction_66',
'Prediction_67',
'Prediction_68',
'Prediction_69',
'Prediction_70',
'Prediction_71',
'Prediction_72',
'Prediction_73',
'Prediction_74',
'Prediction_75',
'Prediction_76',
'Prediction_77',
'Prediction_78',
'Prediction_79',
'Prediction_80',
'Prediction_81',
'Prediction_82',
'Prediction_83',
'Prediction_84',
'Prediction_85',
'Prediction_86',
'Prediction_87',
'Prediction_88',
'Prediction_89',
'Prediction_90',
'Prediction_91',
'Prediction_92',
'Prediction_93',
'Prediction_94',
'Prediction_95',
'Prediction_96',
'Prediction_97',
'Prediction_98',
'Prediction_99',
'Prediction_100',
'Prediction_101',
'Prediction_102',
'Prediction_103',
'Prediction_104',
'Prediction_105',
'Prediction_106',
'Prediction_107',
'Prediction_108',
'Prediction_109',
'Prediction_110',
'Prediction_111',
'Prediction_112',
'Prediction_113',
'Prediction_114',
'Prediction_115',
'Prediction_116',
'Prediction_117',
'Prediction_118',
'Prediction_119',
'Prediction_120',
'Prediction_121',
'Prediction_122',
'Prediction_123',
'Prediction_124',
'Prediction_125',
'Prediction_126',
'Prediction_127',
'Prediction_128',
'Prediction_129',
'Prediction_130',
'Prediction_131',
'Prediction_132',
'Prediction_133',
'Prediction_134',
'Prediction_135',
'Prediction_136',
'Prediction_137',
'Prediction_138',
'Prediction_139',
'Prediction_140',
'Prediction_141',
'Prediction_142',
'Prediction_143',
'Prediction_144',
'Prediction_145',
'Prediction_146',
'Prediction_147',
'Prediction_148',
'Prediction_149',
'Prediction_150',
'Prediction_151',
'Prediction_152',
'Prediction_153',
'Prediction_154',
'Prediction_155',
'Prediction_156',
'Prediction_157',
'Prediction_158',
'Prediction_159',
'Prediction_160',
'Prediction_161',
'Prediction_162',
'Prediction_163',
'Prediction_164',
'Prediction_165',
'Prediction_166',
'Prediction_167',
'Prediction_168',
'Prediction_169',
'Prediction_170',
'Prediction_171',
'Prediction_172',
'Prediction_173',
'Prediction_174',
'Prediction_175',
'Prediction_176',
'Prediction_177',
'Prediction_178',
'Prediction_179',
'Prediction_180',
'Prediction_181',
'Prediction_182',
'Prediction_183',
'Prediction_184',
'Prediction_185',
'Prediction_186',
'Prediction_187',
'Prediction_188',
'Prediction_189',
'Prediction_190',
'Prediction_191',
'Prediction_192',
'Prediction_193',
'Prediction_194',
'Prediction_195',
'Prediction_196',
'Prediction_197',
'Prediction_198',
'Prediction_199',
'Prediction_200',
'Prediction_201',
'Prediction_202',
'Prediction_203',
'Prediction_204',
'Prediction_205',
'Prediction_206',
'Prediction_207',
'Prediction_208',
'Prediction_209',
'Prediction_210',
'Prediction_211',
'Prediction_212',
'Prediction_213',
'Prediction_214',
'Prediction_215',
'Prediction_216',
'Prediction_217',
'Prediction_218',
'Prediction_219',
'Prediction_220',
'Prediction_221',
'Prediction_222',
'Prediction_223',
'Prediction_224',
'Prediction_225',
'Prediction_226',
'Prediction_227',
'Prediction_228',
'Prediction_229',
'Prediction_230',
'Prediction_231',
'Prediction_232',
'Prediction_233',
'Prediction_234',
'Prediction_235',
'Prediction_236',
'Prediction_237',
'Prediction_238',
'Prediction_239',
'Prediction_240',
'Prediction_241',
'Prediction_242'
]

S1=pd.DataFrame(data=None,columns=SS, index=x_train.index)
S2=pd.DataFrame(data=None,columns=SS, index=X_test.index)

i = 0
j = M-1
k = 0

while(j <= 29271):
    list0 = list(range(i,j))
    temp_0 = Train_0.loc[Train_0['counter'].isin(list0)]
    temp = pd.concat((temp_0,Train_1),axis=0)
    temp = temp.sort_values(by=['counter'])
    temp = temp.drop('counter',1)
    
    temp_x = temp.loc[:,temp.columns != 'BecomeTrade']
    temp_y = temp['BecomeTrade']
    #clf = MLPClassifier(hidden_layer_sizes=(25,10),activation='relu')
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(temp_x, temp_y)
    clf_predict_train = clf.predict(x_train)
    clf_predict_test = clf.predict(X_test)
    name = SS[k]
    S1[name] = clf_predict_train
    S2[name] = clf_predict_test
    del clf_predict_train
    del clf_predict_test
    del temp_0
    del temp_x
    del temp_y
    del temp
    k = k + 1
    i = i + 100
    j = j + 100
    
Train_f_x = pd.concat((x_train, S1), axis = 1)
Test_f_x = pd.concat((X_test, S2), axis = 1)

Train_ = pd.concat((Train_f_x,y_train),axis=1)
Test_ = pd.concat((Test_f_x,y_test),axis=1)

Train_.to_csv('xxx/Train_.csv')
Test_.to_csv('xxx/Test_.csv')

# with stacking features, apply AdaBoost directly

clf = AdaBoostClassifier(n_estimators=100)

clf.fit(Train_f_x,y_train)
y_score = clf.decision_function(Test_f_x)

precision5, recall5, _ = precision_recall_curve(y_test, y_score)

predict_train_Stack_AB = clf.predict(Train_f_x)
recall_train_Stack_AB = recall_score(y_train,predict_train_Stack_AB)
precision_train_Stack_AB = precision_score(y_train,predict_train_Stack_AB)
acc_train_Stack_AB = clf.score(Train_f_x, y_train)
f1_train_Stack_AB = f1_score(y_train,predict_train_Stack_AB)



clf_predict = clf.predict(Test_f_x)
clf_proba7 = clf.predict_proba(Test_f_x)[:,1]
clf_fpr7, clf_tpr7, clf_thresholds7 = roc_curve(y_test, clf_proba7)
clf_auc7 = auc(clf_fpr7, clf_tpr7)
recall_test_Stack_AB = recall_score(y_test, clf_predict)
precision_test_Stack_AB = precision_score(y_test, clf_predict)
acc_test_Stack_AB = clf.score(Test_f_x, y_test)
f1_test_Stack_AB = f1_score(y_test, clf_predict)


# Sequential Sampling + Oversampling + AB + Stacked Ensemble

df_f=pd.concat([Train_f_x,y_train],axis=1)

training_0 = df_f[df_f['BecomeTrade'] == 0]
training_1 = df_f[df_f['BecomeTrade'] == 1]


resamplePercent = 0.45
alpha = np.ceil((float(len(training_0))/float(len(training_1)))*resamplePercent)
combined_training = pd.concat([training_1]*int(alpha), ignore_index = True).append(training_0)
    
X2_1 = combined_training.loc[:,combined_training.columns!='BecomeTrade']
df_train_y_1 = combined_training['BecomeTrade']

clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X2_1,df_train_y_1)
y_score = clf.decision_function(Test_f_x)
precision6, recall6, _ = precision_recall_curve(y_test, y_score)
    
predict_train_Stack_AB_Resample = clf.predict(X2_1)
recall_train_Stack_AB_Resample = recall_score(df_train_y_1,predict_train_Stack_AB_Resample)
precision_train_Stack_AB_Resample = precision_score(df_train_y_1,predict_train_Stack_AB_Resample)
acc_train_Stack_AB_Resample = clf.score(X2_1, df_train_y_1)
f1_train_Stack_AB_Resample = f1_score(df_train_y_1,predict_train_Stack_AB_Resample)
#other metrics


clf_predict = clf.predict(Test_f_x)
clf_proba8 = clf.predict_proba(Test_f_x)[:,1]

clf_fpr8, clf_tpr8, clf_thresholds8 = roc_curve(y_test, clf_proba8)
clf_auc8 = auc(clf_fpr8, clf_tpr8)
recall_test_Stack_AB_Resample = recall_score(y_test, clf_predict)
precision_test_Stack_AB_Resample = precision_score(y_test, clf_predict)
acc_test_Stack_AB_Resample = clf.score(Test_f_x, y_test)
f1_test_Stack_AB_Resample = f1_score(y_test, clf_predict)

plt.figure(figsize=(8,10))

#plt.plot([0,1],[0,1], 'k--')
plt.plot(precision2,recall2, label='Raw Data + AB')
plt.plot(precision3,recall3, label='Oversampling + AB')
plt.plot(precision4,recall4,label='Undersampling/50 + AB')
plt.plot(precision5,recall5, label='Sequential Sampling + Stacked Ensemble + AB')
plt.plot(precision6,recall6, label='Sequential Sampling + Stacked Ensemble + AB + Oversampling')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.ylim([0.0,1.05])
plt.xlim([0.18,0.4])
plt.legend(loc='upper left')
plt.show()


plt.figure(figsize=(8,10))

plt.plot([0,1],[0,1], 'k--')
plt.plot(clf_fpr2,clf_tpr2, label='Raw Data + AB')
plt.plot(clf_fpr3,clf_tpr3, label='Oversampling + AB')
plt.plot(clf_fpr6,clf_tpr6, label='Undersampling/50 + AB')
plt.plot(clf_fpr7,clf_tpr7, label='Sequential Sampling + Stacked Ensemble + AB')
plt.plot(clf_fpr8,clf_tpr8, label='Sequential Sampling + Stacked Ensemble + AB + Oversampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower left')
plt.show()

# Undersampling Method

Test_1 = Test.loc[:,Test.columns!='BecomeTrade']
y_test_1 = Test['BecomeTrade']

Test_1.set_index('OrderID',inplace=True)


Train['date']=Train['StartTime'].dt.date

Train.drop(['StartTime'],axis=1,inplace=True)
P=len(Train)

Train = Train.sort_values(by=['date','ContinuousStartTime'])
Train['counter']=range(P)

Train = Train.drop(['date','ContinuousStartTime'],1)


Train.set_index('OrderID',inplace=True)

Train_1 = Train.loc[Train['BecomeTrade']==1]

Q=len(Train_1)

Train_1['sequence'] = range(Q)

Train.shape


a = 0
df = pd.DataFrame()


while (a <= Q-1):
    TEMP = Train_1.loc[Train_1['sequence'] == a]
    Count = TEMP['counter'].values
    
    if (a <= 75):
        list_1 = list(range(0,Count+75))
        temp_0 = Train.loc[Train['counter'].isin(list_1)]
        temp_0 = temp_0.drop('counter',1)
        df = df.append(temp_0)
        del list_1
        del temp_0
    elif (a > 75) & (a <= 34338):
        list_1 = list(range(Count-75, Count+75))
        temp_0 = Train.loc[Train['counter'].isin(list_1)]
        temp_0 = temp_0.drop('counter',1)
        df = df.append(temp_0)
        del list_1
        del temp_0
    elif (a > 34338):
        list_1 = list(range(Count-75, 34413))
        temp_0 = Train.loc[Train['counter'].isin(list_1)]
        temp_0 = temp_0.drop('counter',1)
        df = df.append(temp_0)
        del list_1
        del temp_0
    else:
        continue
    
    a = a + 1
    print(a)
    
    
df.drop_duplicates(inplace=True)


# Mutual information feature selection-forward greedy selection on continous features     
        
ContinousFeatures = [
'Adv20d',
'L1_1',
'L1_5',
'L1_15',
'L1_Open',
'L1_Previous',
'L3_1',
'L3_5',
'L3_15',
'L3_Open',
'L3_Previous',
'x0^2',
'x0 x1',
'x0 x2',
'x0 x3',
'x0 x4',
'x0 x5',
'x0 x6',
'x0 x7',
'x0 x8',
'x0 x9',
'x0 x10',
'x1^2',
'x1 x2',
'x1 x3',
'x1 x4',
'x1 x5',
'x1 x6',
'x1 x7',
'x1 x8',
'x1 x9',
'x1 x10',
'x2^2',
'x2 x3',
'x2 x4',
'x2 x5',
'x2 x6',
'x2 x7',
'x2 x8',
'x2 x9',
'x2 x10',
'x3^2',
'x3 x4',
'x3 x5',
'x3 x6',
'x3 x7',
'x3 x8',
'x3 x9',
'x3 x10',
'x4^2',
'x4 x5',
'x4 x6',
'x4 x7',
'x4 x8',
'x4 x9',
'x4 x10',
'x5^2',
'x5 x6',
'x5 x7',
'x5 x8',
'x5 x9',
'x5 x10',
'x6^2',
'x6 x7',
'x6 x8',
'x6 x9',
'x6 x10',
'x7^2',
'x7 x8',
'x7 x9',
'x7 x10',
'x8^2',
'x8 x9',
'x8 x10',
'x9^2',
'x9 x10',
'x10^2',
'x0^3',
'x0^2 x1',
'x0^2 x2',
'x0^2 x3',
'x0^2 x4',
'x0^2 x5',
'x0^2 x6',
'x0^2 x7',
'x0^2 x8',
'x0^2 x9',
'x0^2 x10',
'x0 x1^2',
'x0 x1 x2',
'x0 x1 x3',
'x0 x1 x4',
'x0 x1 x5',
'x0 x1 x6',
'x0 x1 x7',
'x0 x1 x8',
'x0 x1 x9',
'x0 x1 x10',
'x0 x2^2',
'x0 x2 x3',
'x0 x2 x4',
'x0 x2 x5',
'x0 x2 x6',
'x0 x2 x7',
'x0 x2 x8',
'x0 x2 x9',
'x0 x2 x10',
'x0 x3^2',
'x0 x3 x4',
'x0 x3 x5',
'x0 x3 x6',
'x0 x3 x7',
'x0 x3 x8',
'x0 x3 x9',
'x0 x3 x10',
'x0 x4^2',
'x0 x4 x5',
'x0 x4 x6',
'x0 x4 x7',
'x0 x4 x8',
'x0 x4 x9',
'x0 x4 x10',
'x0 x5^2',
'x0 x5 x6',
'x0 x5 x7',
'x0 x5 x8',
'x0 x5 x9',
'x0 x5 x10',
'x0 x6^2',
'x0 x6 x7',
'x0 x6 x8',
'x0 x6 x9',
'x0 x6 x10',
'x0 x7^2',
'x0 x7 x8',
'x0 x7 x9',
'x0 x7 x10',
'x0 x8^2',
'x0 x8 x9',
'x0 x8 x10',
'x0 x9^2',
'x0 x9 x10',
'x0 x10^2',
'x1^3',
'x1^2 x2',
'x1^2 x3',
'x1^2 x4',
'x1^2 x5',
'x1^2 x6',
'x1^2 x7',
'x1^2 x8',
'x1^2 x9',
'x1^2 x10',
'x1 x2^2',
'x1 x2 x3',
'x1 x2 x4',
'x1 x2 x5',
'x1 x2 x6',
'x1 x2 x7',
'x1 x2 x8',
'x1 x2 x9',
'x1 x2 x10',
'x1 x3^2',
'x1 x3 x4',
'x1 x3 x5',
'x1 x3 x6',
'x1 x3 x7',
'x1 x3 x8',
'x1 x3 x9',
'x1 x3 x10',
'x1 x4^2',
'x1 x4 x5',
'x1 x4 x6',
'x1 x4 x7',
'x1 x4 x8',
'x1 x4 x9',
'x1 x4 x10',
'x1 x5^2',
'x1 x5 x6',
'x1 x5 x7',
'x1 x5 x8',
'x1 x5 x9',
'x1 x5 x10',
'x1 x6^2',
'x1 x6 x7',
'x1 x6 x8',
'x1 x6 x9',
'x1 x6 x10',
'x1 x7^2',
'x1 x7 x8',
'x1 x7 x9',
'x1 x7 x10',
'x1 x8^2',
'x1 x8 x9',
'x1 x8 x10',
'x1 x9^2',
'x1 x9 x10',
'x1 x10^2',
'x2^3',
'x2^2 x3',
'x2^2 x4',
'x2^2 x5',
'x2^2 x6',
'x2^2 x7',
'x2^2 x8',
'x2^2 x9',
'x2^2 x10',
'x2 x3^2',
'x2 x3 x4',
'x2 x3 x5',
'x2 x3 x6',
'x2 x3 x7',
'x2 x3 x8',
'x2 x3 x9',
'x2 x3 x10',
'x2 x4^2',
'x2 x4 x5',
'x2 x4 x6',
'x2 x4 x7',
'x2 x4 x8',
'x2 x4 x9',
'x2 x4 x10',
'x2 x5^2',
'x2 x5 x6',
'x2 x5 x7',
'x2 x5 x8',
'x2 x5 x9',
'x2 x5 x10',
'x2 x6^2',
'x2 x6 x7',
'x2 x6 x8',
'x2 x6 x9',
'x2 x6 x10',
'x2 x7^2',
'x2 x7 x8',
'x2 x7 x9',
'x2 x7 x10',
'x2 x8^2',
'x2 x8 x9',
'x2 x8 x10',
'x2 x9^2',
'x2 x9 x10',
'x2 x10^2',
'x3^3',
'x3^2 x4',
'x3^2 x5',
'x3^2 x6',
'x3^2 x7',
'x3^2 x8',
'x3^2 x9',
'x3^2 x10',
'x3 x4^2',
'x3 x4 x5',
'x3 x4 x6',
'x3 x4 x7',
'x3 x4 x8',
'x3 x4 x9',
'x3 x4 x10',
'x3 x5^2',
'x3 x5 x6',
'x3 x5 x7',
'x3 x5 x8',
'x3 x5 x9',
'x3 x5 x10',
'x3 x6^2',
'x3 x6 x7',
'x3 x6 x8',
'x3 x6 x9',
'x3 x6 x10',
'x3 x7^2',
'x3 x7 x8',
'x3 x7 x9',
'x3 x7 x10',
'x3 x8^2',
'x3 x8 x9',
'x3 x8 x10',
'x3 x9^2',
'x3 x9 x10',
'x3 x10^2',
'x4^3',
'x4^2 x5',
'x4^2 x6',
'x4^2 x7',
'x4^2 x8',
'x4^2 x9',
'x4^2 x10',
'x4 x5^2',
'x4 x5 x6',
'x4 x5 x7',
'x4 x5 x8',
'x4 x5 x9',
'x4 x5 x10',
'x4 x6^2',
'x4 x6 x7',
'x4 x6 x8',
'x4 x6 x9',
'x4 x6 x10',
'x4 x7^2',
'x4 x7 x8',
'x4 x7 x9',
'x4 x7 x10',
'x4 x8^2',
'x4 x8 x9',
'x4 x8 x10',
'x4 x9^2',
'x4 x9 x10',
'x4 x10^2',
'x5^3',
'x5^2 x6',
'x5^2 x7',
'x5^2 x8',
'x5^2 x9',
'x5^2 x10',
'x5 x6^2',
'x5 x6 x7',
'x5 x6 x8',
'x5 x6 x9',
'x5 x6 x10',
'x5 x7^2',
'x5 x7 x8',
'x5 x7 x9',
'x5 x7 x10',
'x5 x8^2',
'x5 x8 x9',
'x5 x8 x10',
'x5 x9^2',
'x5 x9 x10',
'x5 x10^2',
'x6^3',
'x6^2 x7',
'x6^2 x8',
'x6^2 x9',
'x6^2 x10',
'x6 x7^2',
'x6 x7 x8',
'x6 x7 x9',
'x6 x7 x10',
'x6 x8^2',
'x6 x8 x9',
'x6 x8 x10',
'x6 x9^2',
'x6 x9 x10',
'x6 x10^2',
'x7^3',
'x7^2 x8',
'x7^2 x9',
'x7^2 x10',
'x7 x8^2',
'x7 x8 x9',
'x7 x8 x10',
'x7 x9^2',
'x7 x9 x10',
'x7 x10^2',
'x8^3',
'x8^2 x9',
'x8^2 x10',
'x8 x9^2',
'x8 x9 x10',
'x8 x10^2',
'x9^3',
'x9^2 x10',
'x9 x10^2',
'x10^3'
]


df = pd.read_csv('xxx/xxx.csv')
df[:5]
df.set_index('OrderID',inplace=True)

from oct2py import octave

from operator import itemgetter
octave.addpath('xxx/feature_selection/mi/')

Nf=len(ContinousFeatures)

R=ContinousFeatures

L=[]
F={}


vector1 = df['BecomeTrade'].astype(int)
vector1 = vector1.values
vector1 = np.array(vector1).reshape(-1,1)

print(vector1)

for k in range(Nf):
    W={}
    Nr=len(R)
    for i in range(Nr):
        X_col = R[i]
        vector2 = df[X_col].values.reshape(-1,1)
        print(vector2)
        condvec = df[L].values
        print(condvec)
        score = octave.condmutualinfo(vector2,vector1,condvec)
        W[X_col]=score
    SortedValue = sorted(W.items(),key=itemgetter(1),reverse=True)
    
    winner = SortedValue[0]
    key = winner[0]

    L.append(key)
    F[key]=winner[1]
    R.remove(key)
    del score
    del SortedValue
    del X_col
    del vector2
    del condvec
    del winner
    del key
    


rank_cont = pd.DataFrame.from_dict(F.items())
rank_cont.to_csv('xxx/rank_cont.csv')

import itertools
from collections import OrderedDict

F3 = OrderedDict(sorted(F.items(),key = lambda x: x[1],reverse=True))
F2 = dict(itertools.islice(F3.items(),10))

names = list(F2.keys())
values = list(F2.values())
plt.figure(figsize=(10,5))
plt.bar(range(len(F2)), values, tick_label=names)
plt.title('mutual informaiton ranking')
plt.show()


# Feature selection for categorical features
    
    
CategoricalFeatures = []

from cModPyDem import *

import sys

sys.path

sys.path.append('xxx/mrmrPy/run/')


"""
# Usage: mrmr_osx -i <dataset> -t <threshold> [optional arguments]
#	 -i <dataset>    .CSV file containing M rows and N columns, row - sample, column - variable/attribute.
#	 -t <threshold> a float number of the discretization threshold; non-specifying this parameter means no discretizaton (i.e. data is already integer); 0 to make binarization.
	 -n <number of features>   a natural number, default is 50.
	 -m <selection method>    either "MID" or "MIQ" (Capital case), default is MID.
	 -s <MAX number of samples>   a natural number, default is 1000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
	 -v <MAX number of variables/attibutes in data>   a natural number, default is 10000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
	 [-h] print this message.


 *** This program and the respective minimum Redundancy Maximum Relevance (mRMR) 
     algorithm were developed by Hanchuan Peng <hanchuan.peng@gmail.com>for
     the paper 
     "Feature selection based on mutual information: criteria of 
      max-dependency, max-relevance, and min-redundancy,"
      Hanchuan Peng, Fuhui Long, and Chris Ding, 
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      Vol. 27, No. 8, pp.1226-1238, 2005.

# Modified by Zhiyuan Yao on 03/02/2018.
# Important Note: MaxRel() and mrmr() can't be run at the same time. 
# Return type: dictionary
"""

# Example:
Inpt = './mrmr -i ./LEVL_first_hour_result.csv -t 2'

#print('MaxRel',MaxRel(Inpt.split()))
#print('mrmr',mrmr(Inpt.split()))
a = mrmr(Inpt.split())
print('score of L3_one_min: ', a['L3_one_min'])

"""
Catedata = Combined[['BecomeTrade','Symbol','VenueType','Side','SecurityCategory','Sector','MktCap']]


Catedata.dtypes

Catedata['Symbol'] = Catedata['Symbol'].astype('category')

Catedata['Symbol_cat'] = Catedata['Symbol'].cat.codes

Catedata['VenueType'] = Catedata['VenueType'] .astype('category')

Catedata['VenueType_cat'] = Catedata['VenueType'].cat.codes

Catedata['Side'] = Catedata['Side'] .astype('category')

Catedata['Side_cat'] = Catedata['Side'].cat.codes

Catedata['SecurityCategory'] = Catedata['SecurityCategory'] .astype('category')

Catedata['SecurityCategory_cat'] = Catedata['SecurityCategory'].cat.codes

Catedata['Sector'] = Catedata['Sector'] .astype('category')

Catedata['Sector_cat'] = Catedata['Sector'].cat.codes

Catedata['MktCap'] = Catedata['MktCap'] .astype('category')

Catedata['MktCap_cat'] = Catedata['MktCap'].cat.codes

Catedata_1 = Catedata[['BecomeTrade','Symbol_cat','VenueType_cat','Side_cat','SecurityCategory_cat','Sector_cat','MktCap_cat']]

Catedata_1.to_csv('xxx/Catedata_1.csv',index=False,index_label=False)

Catedata_1 = pd.read_csv('xxx/Catedata_1.csv')

Catedata_1['BecomeTrade_cat']=Catedata_1['BecomeTrade'].astype(int)

Catedata_1 = Catedata_1.drop(['BecomeTrade'], axis=1)

cols = list(Catedata_1)

cols.insert(0,cols.pop(cols.index('BecomeTrade_cat')))

Catedata_2 = Catedata_1.ix[:,cols]

Catedata_2.to_csv('xxx/Catedata_2.csv',index=False,index_label=False)

"""