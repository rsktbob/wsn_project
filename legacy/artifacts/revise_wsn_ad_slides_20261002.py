"""Rebuild only A-D slide XML; preserve all other PPTX ZIP entries byte-for-byte."""
from pathlib import Path
import zipfile,xml.etree.ElementTree as E,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
SRC=Path('/home/g114056067/.codex/attachments/4c27e225-df08-4c4c-8786-1d6afb5d5c36/wsn報告2.pptx')
OUT=ROOT/'wsn問題研究/thesis/wsn報告2_A-D修訂_20261002.pptx'
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
E.register_namespace('p',P);E.register_namespace('a',A);E.register_namespace('r',R)
def node(parent,ns,name,attrs=None):return E.SubElement(parent,'{'+ns+'}'+name,attrs or {})
def emu(x):return str(round(x*914400))
def box(tree,id,name,x,y,w,h,paragraphs,size=22,color='23354D',bold=False):
 sp=node(tree,P,'sp');nv=node(sp,P,'nvSpPr');node(nv,P,'cNvPr',{'id':str(id),'name':name});node(nv,P,'cNvSpPr',{'txBox':'1'});node(nv,P,'nvPr')
 prop=node(sp,P,'spPr');xf=node(prop,A,'xfrm');node(xf,A,'off',{'x':emu(x),'y':emu(y)});node(xf,A,'ext',{'cx':emu(w),'cy':emu(h)})
 geo=node(prop,A,'prstGeom',{'prst':'rect'});node(geo,A,'avLst');node(prop,A,'noFill');ln=node(prop,A,'ln');node(ln,A,'noFill')
 tx=node(sp,P,'txBody');bp=node(tx,A,'bodyPr',{'wrap':'square','lIns':'0','rIns':'0','tIns':'0','bIns':'0','anchor':'t'});node(bp,A,'noAutofit');node(tx,A,'lstStyle')
 for text in paragraphs:
  para=node(tx,A,'p');pp=node(para,A,'pPr',{'marL':'0','marR':'0','indent':'0','algn':'l'});node(pp,A,'buNone');spacing=node(pp,A,'lnSpc');node(spacing,A,'spcPct',{'val':'112000'});after=node(pp,A,'spcAft');node(after,A,'spcPts',{'val':'1400'})
  run=node(para,A,'r');rp=node(run,A,'rPr',{'lang':'zh-TW','sz':str(size*100),'b':'1' if bold else '0'});fill=node(rp,A,'solidFill');node(fill,A,'srgbClr',{'val':color});node(rp,A,'latin',{'typeface':'Arial'});node(rp,A,'ea',{'typeface':'Microsoft JhengHei'});node(run,A,'t').text=text
  node(para,A,'endParaRPr',{'lang':'zh-TW','sz':str(size*100)})
 return sp
slides={2: ('A. Attention｜研究動機', ['無線感測網路需在有限電量下持續覆蓋目標，並透過可行路由把資料送達基地台。', '節點啟用、感測範圍與轉送負載互相影響；局部省電不必然讓整體網路活得更久。', '本研究關注：如何有效使用有限搜尋預算，找到兼顧覆蓋、連通及能源使用的排程與路由方案。'], '[R2] Chen et al., 2020, doi:10.1007/978-981-15-6113-9_13；[R6] Kabakulak, 2018, arXiv:1802.10438。'), 3: ('B. But｜從前作建立研究問題', ['既有方法：SE 依子空間潛力分配搜尋資源；SA-SETS 已依節點狀態選擇識別感測器，改善固定分區失效的問題。[R1–R3]', '待釐清的機制：識別基因所定義的區域，不必然對應解碼後的感測啟用特徵；區域分類也不直接指定哪些感測器應一起修改。', '研究問題：在相同編碼與解碼規則下，以感測器分段引導操作，能否改善有限預算下的有效探索及網路壽命？改善仍待驗證。'], '[R1] Tsai, 2015, doi:10.1109/SMC.2015.447；[R2] Chen et al., 2020；[R3] 洪以恩碩論, 2023, §3.3。'), 4: ('C. Cure｜Ring_SETS', ['保留「感測等級＋優先權」編碼及覆蓋資訊輔助初始化，改以感測器序列片段決定交配與突變範圍。', '全體搜尋者使用共享商品池，結合段內／跨相鄰段交換，以及角色分離的菁英更新。', '以 child1 品質與 Beta-CDF 引導選段；透過消融分開驗證操作範圍、共享池及更新方式的效果。'], '研究對象：Algorithm/se/Ring_SETS.py。[R5] Omidvar et al., 2014 支持檢查分組與變數交互作用；不含 RL。'), 5: ('D. Development｜整體資料流', ['① 讀取座標、能源與覆蓋表；依 BS 物理環及角度排列感測器，建立操作片段。', '② 以 CCS 輔助產生 [level, priority] 候選；解碼並評分，建立搜尋者與共享商品池。', '③ 選片段 → 與商品交換 → 在同範圍突變 → 評估兩種角色子代 → 分別接受改善。', '④ 更新片段品質與 Beta-CDF，繼續搜尋；最後以完整生命周期檢驗覆蓋、連通、耗能及壽命。'], '實作：Ring_SETS.py、SensorEncoding.py；固定解碼器與 fitness 版本，避免比較時混入其他変更。'), 6: ('D. Development｜感測等級與優先權', ['100 顆感測器共有 200 維：x = [L₁, P₁, L₂, P₂, …, L₁₀₀, P₁₀₀]。', '預設 L 為 0–5 的感測選項索引，對應半徑 [0, 3, 6, 9, 12, 15]；0 表示關閉。P 為 0–9 的優先權基因。', 'SensorEncoding 解碼會做覆蓋挑選、冗餘移除及路由建立，因此基因指定的 level 不一定全部保留到實際排程。', 'Ring 與 SA 使用相同基因域；研究差別在操作與更新機制，不能宣稱 Ring 讓原始解空間變小。'], '依目前 Ring_SETS 與 SensorEncoding（預設 v3）。實際合法感測選項仍須隨實驗設定記錄。'), 7: ('D. Development｜四段操作的意義', ['原 SETS：以識別感測器的開關基因組合標記解空間區域，商品依區存放。', 'Ring_SETS：四段決定這次可以修改哪些感測器；所有候選解都描述完整網路，各段使用同一商品池。', '100 顆、h=4 時每段 25 顆。80% 選段內連續子片段；20% 選整個片段加相鄰片段的一部分。', '選段是選操作範圍，不強制感測器開或關。child1 更新搜尋者，child2 競爭更新商品。'], '實作：Ring_SETS 的 select_sensor_span、make_children、align_region、vision_search。'), 8: ('D. Development｜空間排序的假設', ['先依 BS 距離分物理環、同環按角度排序，再依序列切成近等數量的操作片段。', '片段可能跨物理環或切開同一環；四段不等於固定四個半徑範圍。靠近 BS 的節點也不保證總是轉送熱點。', '環狀排序是 Problem 共用前處理，SA 組也可能已啟用。因此須另比純距離及固定隨機排序。', '驗證共同覆蓋／共享路由、實际子代變化與相同預算下的改善；不預設跨地圖角色一致。'], '[R5] 變數交互作用提供分組研究依據；ring 分段是否優於其他排序需本題實驗。'), 9: ('D. Development｜CCS 初始化與限制', ['先隨機產生 level 與 priority；再對每個 target 從可覆蓋候選中抽取 (sensor, level)，寫入感測等級基因。', '目前 SA_SETS 與 Ring_SETS 在分區對齊前使用相同 CCS 初始化；它是共同基礎，不列為 Ring 的新增貢獻。', '不同 target 可能選到同一 sensor；後次寫入可覆蓋先前 level，因此不保證初始化達到 100% 覆蓋。', '覆蓋、路由、能源可行性分別檢查；初始化改善與操作機制的效果，應以獨立消融驗證。'], '實作：SA_SETS.create_candidate 與 Ring_SETS.create_candidate；對應初步診斷見 Ring_SETS 修訂報告。')}
with zipfile.ZipFile(SRC) as src,zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as dst:
 for item in src.infolist():
  data=src.read(item.filename)
  index=next((i for i in slides if item.filename==f'ppt/slides/slide{i}.xml'),None)
  if index is not None:
   title,paras,foot=slides[index]
   slide=E.Element('{'+P+'}sld');cs=node(slide,P,'cSld');bg=node(cs,P,'bg');bgp=node(bg,P,'bgPr');fill=node(bgp,A,'solidFill');node(fill,A,'srgbClr',{'val':'FFFFFF'});node(bgp,A,'effectLst')
   tree=node(cs,P,'spTree');nv=node(tree,P,'nvGrpSpPr');node(nv,P,'cNvPr',{'id':'1','name':''});node(nv,P,'cNvGrpSpPr');node(nv,P,'nvPr');gp=node(tree,P,'grpSpPr');xf=node(gp,A,'xfrm')
   for n,at in [('off',{'x':'0','y':'0'}),('ext',{'cx':'0','cy':'0'}),('chOff',{'x':'0','y':'0'}),('chExt',{'cx':'0','cy':'0'})]:node(xf,A,n,at)
   box(tree,2,'Title',.55,.35,12.15,.75,[title],30,'123A57',True)
   box(tree,3,'Body',.65,1.4,11.95,4.85,paras,22)
   box(tree,4,'Sources',.65,6.55,11.95,.65,[foot],10,'526578')
   cm=node(slide,P,'clrMapOvr');node(cm,A,'masterClrMapping')
   data=E.tostring(slide,encoding='utf-8',xml_declaration=True)
  dst.writestr(item,data)
# Validate all XML; only the 8 targeted slides may differ.
with zipfile.ZipFile(SRC) as a,zipfile.ZipFile(OUT) as b:
 assert b.testzip() is None
 changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 assert set(changed)=={f'ppt/slides/slide{i}.xml' for i in slides},changed
 for n in b.namelist():
  if n.endswith(('.xml','.rels')):E.fromstring(b.read(n))
 for i in [1,10,11,12]:assert a.read(f'ppt/slides/slide{i}.xml')==b.read(f'ppt/slides/slide{i}.xml')
report={'source':str(SRC),'output':str(OUT),'changed_parts':changed,'EF_slide_xml_identical':True,'zip_and_xml_valid':True,'visual_rendering':'not performed: no PowerPoint or LibreOffice renderer installed','sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}
(ROOT/'artifacts/wsn_ad_pptx_validation_20261002.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
