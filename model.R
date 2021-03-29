


r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)


# install_packages <- function() {
#   install.packages("caret")
#   install.packages("xgboost")
#   install.packages("e1071") #for xgboost classification
#   #install.packages("gbm")
#   install.packages("rpart")
#   install.packages("dplyr")
#   #install.packages("tidyverse")
#   #install.packages("doBy")
#   #install.packages("patchwork")
#   #install.packages("hrbrthemes")
#   #install.packages("doParallel")
#   install.packages("statmod")
#   install.packages("tweedie")
#   install.packages("tidyr")
#   #install.packages("cowplot")
#   #install.packages("randomForest")
#   #install.packages("Metrics")
#   #install.packages("skimr")
#   install.packages("tibble")
#   install.packages("tictoc")
#   #install.packages("mice") #to impute values for vh_weights with 0
# }
# 
# install_packages()

# LIBRARY INIT ####

#library(ggplot2)
library(lightgbm)
library(tidyverse)
library(dplyr)
#library(doBy)
#library(hrbrthemes)
library(caret)
library(caretEnsemble)
#library(gbm)
library(xgboost)
library(catboost)
library(glmnet)
library(ranger)  #for RandomForest
library(kernlab) #for SVM
library(e1071)
library(doParallel)
library(tidyr)
#library(randomForest) # for our model
#library(Metrics) # handy evaluation functions
library(statmod)
library(tweedie)
#library(cowplot)
#library(skimr) #to print nice tables
library(tibble) #to load veh_groups
#library(MASS) #for stepaic    #must be loaded after dplyr
#library(mice)  #to impute weights
#library(doRNG) # Reproducible parallelization
library(tictoc)
library(Metrics)
library(broom)
library(plotmo)
library(skimr)
library(mgcv)



add_infl_factor <- function(df){
  
  # Inflating claim_amount to current $
  # 1.010846, 1.021810, 1.032892, 1.044095
  
  from_1_to_4 <- (1.044095 / 1.010846)
  from_2_to_4 <- (1.044095 / 1.021810)
  from_3_to_4 <- (1.044095 / 1.032892)
  from_4_to_4 <- 1
  from_5_to_4 <- 0.975
  
  year <- c(1,2,3,4,5)
  infl_factor <- c(from_1_to_4,from_2_to_4,from_3_to_4, from_4_to_4, from_5_to_4)
  
  infl_mapping <- data.frame(year,infl_factor)
  infl_mapping
  
  df <- left_join(df, infl_mapping, by = "year")
  
  return(df)
}

strip_glm <- function(cm) {
  cm$y = c()
  cm$model = c()
  
  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$qr$qr = c()  
  cm$linear.predictors = c()
  cm$weights = c()
  cm$prior.weights = c()
  cm$data = c()
  
  
  cm$family$variance = c()
  cm$family$dev.resids = c()
  cm$family$aic = c()
  cm$family$validmu = c()
  cm$family$simulate = c()
  attr(cm$terms,".Environment") = c()
  attr(cm$formula,".Environment") = c()
  
  cm
}

#https://stats.stackexchange.com/questions/102667/reduce-random-forest-model-memory-size/171096#171096
strip_rf <- function(cm) {
  cm$finalModel$predicted <- NULL 
  cm$finalModel$oob.times <- NULL 
  cm$finalModel$y <- NULL
  cm$finalModel$votes <- NULL
  cm$control$indexOut <- NULL
  cm$control$index    <- NULL
  cm$trainingData <- NULL
  
  attr(cm$terms,".Environment") <- c()
  attr(cm$formula,".Environment") <- c()
  
  cm
}



golden_features <- function(){
  
  load("f_claims.RData")
  
  return(f_claims)
}

f_golden <- golden_features()



vh_groups <- function(){
  
  
  vhmm_mapping <- tribble(
    
    ~vh_make_model,~vh_make_model_enc,
    "aawqanlavsjfqrne",97.5364679134019,
    "aaykjdjgdzrrdvxz",107.165092271094,
    "abacekzzrkhtgpcp",101.961191105754,
    "abcepdrvvynjsufa",100.046943366716,
    "abipwhwqnzenjxfn",105.444003443812,
    "ablxjgbyowxrfxed",110.683394899993,
    "aceqpjprqgzhffuw",104.80695721254,
    "acvypvzmenxkevbm",102.728362939846,
    "adgfkcvmsaxxghoc",97.7606658795177,
    "adhoqfsfdpetomvs",96.2694335090921,
    "admgymnmeilfhmji",111.443953512246,
    "adzzjitkyqlberpu",101.294469979628,
    "aewtczgiyochvagl",117.300507962131,
    "aewtdnpoiopumymt",102.290054460895,
    "afmufyguudlwbcix",114.577684641055,
    "aggyqhwjksgqtxdd",111.251999245413,
    "agowcnternxraavr",100.102788117016,
    "aifsqdniwqmcuqpv",101.041375936793,
    "aivacsqryguqpdib",104.59352025754,
    "ajdmkzcduerbdsww",104.332114329475,
    "ajktbllxjzfdtwpy",106.269005837038,
    "ajtardhciglimsdi",99.7345540149497,
    "ajvhjkzguyeszaqp",110.67437110914,
    "akqknybjyxwbdpot",108.757420635085,
    "aloltvlyufzyxfvg",109.11143053502,
    "alrfnehgsdtsunhm",100.841694410157,
    "ambmbeydwsdljdcc",101.913159532926,
    "ammbrasbxojlitmt",99.5021079845266,
    "anqyvxqouldudiww",101.794396306653,
    "anrwlguztftzfdng",101.611705898534,
    "anwpfxivfvhnobvz",111.78194417608,
    "aocfhyagfzdywcih",98.4427997589001,
    "aoytjdcfreqvurza",113.343359152447,
    "aparvvfowrjncdhp",86.6017493227109,
    "arcsdpohuzvikyaw",102.756263673648,
    "arfkjhowhuqewzvc",102.133745446607,
    "arfzuuojdtlgxehv",102.534011145241,
    "asbtrxjnhqdpazot",114.096983507556,
    "asmpttrlkodaejic",103.921196766882,
    "atsglyxkfbaztzlj",108.240323047019,
    "avrwlknteymnpjpk",110.355335484577,
    "ayeiibefzqqbyksg",107.406930214086,
    "aysnuezuqgjioyyf",99.0182945543871,
    "aywlgifrijfokyzu",98.5655882455661,
    "azxtekfvyycfmnpt",108.803313279632,
    "baqjsealekltnrgg",111.324664439562,
    "bawsoqdugnynetyj",110.031877141394,
    "bdmklueoovgkajff",105.289052827986,
    "beagnicqcxahqkeq",101.403089106654,
    "bfmdeosllvjkezwq",107.177526805858,
    "bfmvfelwblrzqfyr",104.55408721726,
    "bgbhznmwwidntzab",102.044084750929,
    "bgqrpfiflzijywyu",106.89255940288,
    "bikffjqejohkyhat",103.754904376723,
    "biqzvbfzjivqmrro",93.4737340951363,
    "bkwszkqrqybfgpyn",106.134787693919,
    "blcuqlgntjavsyhs",101.741208018189,
    "blmjcblhzfqwhgew",108.128371185376,
    "bnvgzfegimthyhyo",102.370056988419,
    "bowuhkfextvyabch",117.621827300052,
    "bpqxbrvavqshzebb",109.23735575184,
    "bpuzzsqfyvebjzjg",107.79708906731,
    "bqcnaxkvbmfieysy",100.843355286742,
    "brjgjnnpueqkyaxo",112.785790075122,
    "bsiyfrkwdyptmwji",99.9287552356451,
    "btjxvrgfduskmpts",107.708989015651,
    "buuihjqtdgilqzjc",107.493945069703,
    "bvfbihgnteuiuaov",106.563007023251,
    "bvkytcvosbaunupg",101.871373023847,
    "bvuzvpriwqlnbjxt",101.924819525994,
    "bwjkokfezucsuigb",121.354662412782,
    "bwpieeluivljdtai",103.809233908515,
    "bxksiwcqwmxjcbci",110.830077593927,
    "bxrkvmsmoqvefhra",106.060759048347,
    "bxzfdlphpiwyjeys",110.909183290271,
    "byvoguptigfevpyy",99.2021922575094,
    "bzsxlzwfqbnmljsm",104.170144006517,
    "caovvakxarqpgymh",114.503260195553,
    "cazrxylvhylncoze",104.519582366949,
    "cbmmnlpqoyyursux",102.99861267515,
    "ccxwaznvwtdltwlt",108.003591045704,
    "cedczcxvthqqkwvn",109.555548293655,
    "cedgzkylsgxnlcjg",107.001593518591,
    "celpzeaubkxaxxbx",108.839282955546,
    "ceswaufhjtmqcndn",110.491251789251,
    "cfuyjykoohewxzeg",108.374155266753,
    "cgighhnwnkxluccz",106.264430288807,
    "cgkclpnidlmetsrb",101.913492917881,
    "cgrdxjyaxssrszjz",104.515732079893,
    "cictcfpmfdmknnye",106.921131529746,
    "ciuxczxwhwbxdkdf",108.833629250075,
    "cjcthmigqkejxuzi",103.986596543158,
    "ckxqqcnqrqxijmmf",109.947051136209,
    "clcqzivttlcdfpnv",114.995333219389,
    "cllupxtcyclounsg",109.292265711794,
    "clmcokjtplrbzvuh",100.74374481731,
    "clsrzyechukbaeat",99.9590305008862,
    "clwswcgzlaojjddv",100.758833796921,
    "cmjjolnwfprpzntz",100.48990418077,
    "cmmuslxsfluvfyof",108.112066069982,
    "cnicorpxweynumqk",104.924973803702,
    "cnlvybtdupkcwczn",104.136745550043,
    "cnvpgiyrcrbsvtxo",111.989555138267,
    "coufviypetbrtevy",99.7549298753412,
    "cpixpqtyjwdgmldj",116.983729441308,
    "cpruzckbhhcyorgf",110.185742022415,
    "cqdmtwkacajclcml",100.625397028668,
    "cqewccykrcmvawlo",107.398162879573,
    "csxjshhnfbtgjcgm",100.269871134289,
    "ctachoeiozcpkmst",102.094200103156,
    "cufklbvsirnawzmv",105.696172234243,
    "cuxaapvakeemmbaa",107.367607524658,
    "cwrigmmyfzesuezf",110.925047708585,
    "cwshqcgmaazzefkx",106.171271674417,
    "cwxtybsrimchiwdv",106.019846561332,
    "cxvltpchlhlatjkb",106.580313567993,
    "cxxzogxxkmkjwqui",98.9580711139605,
    "cyftaexytlgvmcbd",108.775632010301,
    "dbtkrhmbfxpkqbau",98.0094673178742,
    "dcjzdpoxqvgnjpmi",103.108602527676,
    "degdppvcniqrzruc",114.463853109318,
    "degvuccboupdnasm",100.169546774323,
    "demgvtbzilochupd",98.9365549830221,
    "dgwbxitzfzbegnoc",100.790489082289,
    "dgwtezteqyzzylho",99.0386748738623,
    "dhjmmmtnpcnalzna",102.339985677627,
    "dhxftxnxtxlgqcqb",103.604537180733,
    "disoykeofihapsal",110.194758997669,
    "djxdgbpuyerxgrmx",109.512766562417,
    "djyptluftbfkxtjd",102.134023522104,
    "dkgrgmlhhtnvzmps",105.174505555518,
    "dlbnpwopifytzerl",104.731728450498,
    "dlemjwpmokwptnai",110.035985064212,
    "dllcylnkzeegtsgr",89.8211962806785,
    "dlrodwgixwmoquny",103.418295528466,
    "dlrvgwmumnwcjixm",102.523259744062,
    "dluodrxtjdtvbxug",98.8668286466254,
    "dlwcludeemsmffyb",107.786778103131,
    "dmqhptvycdmkaxbw",103.06671055533,
    "dnvcqpxxzahdhbvy",103.884071530405,
    "dohofttmidfqjozb",102.307703093795,
    "dohuwjuguzyvqaqg",108.540099750731,
    "doohwubeqhbkevhr",97.2035974726818,
    "dpcnodgqfivkhxvn",97.6223592635136,
    "dpklliwcxycpfriu",99.4138521548935,
    "dqgtaigmpivatpeu",103.654433195123,
    "dqmsefrpxrwielmk",101.979951081298,
    "dqqtizjjhjmqdqqb",107.145271034761,
    "dqxenajfgcimjgnw",101.149155607383,
    "drptidaltxzxopwv",108.31196189356,
    "drpwkafcvcypyrmw",105.323547116369,
    "dsqmtbudvjtnnjwq",104.883032055851,
    "dtdrfrtruyhvbztx",115.599178823637,
    "dtpbahjtnmyuxqno",90.9298412133653,
    "dvmnbbkcvcgwnaen",106.418271865459,
    "dvshwarqhxfcgwfd",113.641025090685,
    "dweqmfoluivgiayj",109.708939035802,
    "dwhlbcevejvegsob",100.945754311967,
    "dwsasdexwmpsmowl",93.2179416292479,
    "dxoirhatawazqmey",112.340220840554,
    "dxpafctvukcmaqao",105.174060954325,
    "dychjlsxfaurgode",111.070968040066,
    "dyzvyrmcdyybbddd",101.226811828002,
    "dzbwjjmruyqxyvms",120.031401763125,
    "dzjyqrdmawtdcqbx",104.380810368015,
    "ebdcmhmtqnfkaalo",102.427878210954,
    "edlxghhjgpmvhabz",103.584314950769,
    "efhjvgwyjfjqsdna",106.038741477244,
    "efiskxgaocgqqjvr",101.241359762134,
    "efyukbppkfgttvvw",108.379453503215,
    "egsfpimnisvvfkne",126.443017084254,
    "ehapkksqqcbofeid",101.276176454245,
    "ehtbxdjhvcwdapsg",101.808605242007,
    "eivjhovgfnfctgjy",99.9001972881713,
    "ejbxcyhffvcouoxd",101.399980818778,
    "ejeggxbwhufjtjhd",101.667643529272,
    "ejlwzigdhipvpndt",106.828053717082,
    "ekwqttgkaobektch",116.339144101504,
    "ekztjicqomhuclqr",105.090305448592,
    "elvboiqxkxwhtgzg",109.929734473621,
    "enmicxqiumbpozpk",109.738700360293,
    "eokuiduvnrtzavmr",104.500203105351,
    "eootycnsxmeekotz",107.922488099146,
    "epbwnmcyogpybxlm",97.6493101001692,
    "erouvyhobhzcycuk",105.700365646287,
    "esilvarzflhfmjhh",96.7141305608427,
    "esiuazjovwvdlgjy",106.88865936624,
    "eslneidrjqwzpqhd",100.3827310511,
    "esvszhlxzbxeecme",108.212945998153,
    "etegzqakpcvyhkaj",99.8943585663871,
    "ettwalwfkzvwdasa",97.5325392314727,
    "eudwptcohxaazhpt",92.2595108563772,
    "evuqnfndofizyoqn",103.948313979568,
    "ewkcexkqpsyfnugi",98.7985130638174,
    "exfftzvkfnajarkm",110.238101547114,
    "exkqtrkthhgvjqdl",100.613357050444,
    "exutskjkecvotaxd",107.159662231486,
    "eyaqhofitsegmcwi",114.14208763726,
    "eyrukxfjgrcdrqeo",108.388646130983,
    "eyrwkwxecpzxzscp",97.1743437853544,
    "ezaffjpqpacrufvd",106.214347649948,
    "ezjnsjxvhnocwwix",106.72856532157,
    "fadjogsnmecatcfb",95.1696579490926,
    "fbvdqkwltwgykywc",119.860236534412,
    "fdbwfjqkichwdebq",105.77132322192,
    "feioipyfbkxhcsyq",114.882423516019,
    "fgxxyxcbjkodwcln",102.515831073453,
    "fhliexbdvrlrpjvx",111.944529555183,
    "fijtohsiakkeuuct",120.405250347045,
    "fjimpbebyszdttpl",105.438520568323,
    "fjzqkqcjerkjykkk",104.874395686815,
    "fklewvbxuecmupxn",96.0608230391152,
    "fkltkgzmjnzqzlqv",107.220892287955,
    "fliymzbupomtmyry",103.242730146795,
    "flnipmkwonjnaqsp",112.000284010653,
    "flpmjcetsinyjimc",104.142078213671,
    "fnfpmchfyyqmdtfm",111.390861229562,
    "fnqgfjfkzhfbiicl",103.638784088442,
    "fouvkndsdstwjqpj",99.3000254824273,
    "fozvmjndontqoxpg",104.240250796183,
    "fpfzaadmykntrupr",102.1356392105,
    "frdityocokfyohoa",109.620622830622,
    "frvooqzltrzlbhxb",108.205767814359,
    "fszxbpjtsihsmnqv",105.216690439615,
    "fuddhlszptfmosir",101.188007425337,
    "fupocenmkiiluzpe",108.907688324163,
    "fuwhdjmdexrstmmo",100.621940333312,
    "fvflhdedljqrcqle",109.284538573158,
    "fvrvkxucfyuyfpbk",104.719191926262,
    "fvsyahnxhitfllgt",100.611136729491,
    "fwqrokhhbukfpssj",112.386463252955,
    "fydlanmzkobgcfsj",106.844906018389,
    "fzcjreusldmxavjy",109.261313114028,
    "fzgvfpmdmggikezp",114.405914885643,
    "gapclpflkdsbeorm",97.6544339371055,
    "gbkevbmczkqhkmoc",110.988180491949,
    "gcmwblighdilwauf",106.210209571667,
    "gctieesvmkeoozqx",105.613426138808,
    "gdaxhrlhuilhiijt",94.5268266986182,
    "gdtzpvajphaxanpi",110.964549414256,
    "gdultxlilvdnuwso",112.661574814001,
    "gdzfmtghobzpihgc",106.709365442299,
    "gfhjqtkgvomiygvx",96.44495002472,
    "gforiqpfasfwlkfl",104.216874152971,
    "ggadbhlnfgoflkaf",100.454058749905,
    "ggaokfjtqxyctvok",102.430664084977,
    "ggidexivtrafqwem",95.2002029662715,
    "ggqsqgrasnpkxano",106.235426982219,
    "gguphuccgeqyojbl",106.347297129175,
    "ggzcspiycgszcunf",108.334393134219,
    "gicokqmbjnafngon",105.65798321024,
    "giyhzprslgbwsaeu",107.362896284446,
    "gjbalugsikirqoam",91.9042607315724,
    "gjblfwqtnckjletn",105.185595621972,
    "gjchrdhbeixppooh",97.774628245544,
    "gjpgirzuabhfpkjd",99.3215806993125,
    "gjxmrfgnorpfspbb",100.734539902812,
    "gkniccewzkphqzrp",107.297512378671,
    "gkxcvooedomgcagl",109.192563357769,
    "gmzbnaysqjpkzqbt",100.398591750682,
    "godkpvbnbdeseoct",100.93196972469,
    "goropquvqaoaajrk",105.886060026184,
    "gpclrtlzecazeeev",101.020800750117,
    "gqfadgvnztixxbmv",100.427790285149,
    "grnnfnsjjydskrht",101.818823812397,
    "grpzbvvgujnswyyg",104.652243539408,
    "gsbeyysssgzgkkuo",101.949855500965,
    "gsooyxmnwsucrksh",99.161261508203,
    "gspurupoewenqznk",99.7986537388292,
    "gtbzqhsuzzdfhzfv",105.306576307003,
    "gtvhxebtkefavzhg",108.862526986116,
    "guiimarisyyjqnfg",98.5523832047496,
    "gujwvdfcmmqcwxfi",96.5621597324848,
    "gvaasolsbmnbjhah",105.383721699255,
    "gvordmjbkxszftsl",100.397394766296,
    "gvsbsfrfcvftmytm",108.17803134623,
    "gvxirlwrjrrnoadg",99.8670057803599,
    "gwptulznqgygeegy",106.618041211432,
    "gxgjyxrnnugizdvf",106.856552240388,
    "gxpuiivthwcmpcmc",104.290262534717,
    "gyhebbdhtmqwwxnp",100.667095930863,
    "gykwyopsdhbsalvd",114.837951757601,
    "gzebcnjcmqioqcjb",93.9553697468495,
    "gzpmemdiurffxomf",102.811215518997,
    "haowzcsrftoqsrvi",109.376841561416,
    "hayciibjzwapccnb",109.824188318211,
    "hcfpedolsygjsofb",99.9356217788945,
    "hcoxxbfccserxklx",105.871938882484,
    "heicadwqfavetjwx",113.322730404037,
    "hgyoclvrybybkocm",106.968895493982,
    "hhidavhckwcwznhf",112.607967510064,
    "hhrmdevbfqiebnum",102.516057829179,
    "hikofhdgvhuwkixj",105.737819803226,
    "hixbnwflcimyepla",112.834942883336,
    "hjejiuqyfrvtxagi",110.773850574719,
    "hjhlpxkdgqzdlnkc",113.336508502722,
    "hjhvhzfpslejsnej",102.621764768948,
    "hjyumbyuzbeubtbb",104.806051468581,
    "hkazsxqvbtmawovu",100.695804011649,
    "hotinomqpajebeov",109.355146318569,
    "hpohizpkyzvwunni",93.9529267374161,
    "hqixaqcgdcbagrmw",108.589659615173,
    "hrlyreijarvikmlk",110.380861427623,
    "hruelqcyvmwzsqkp",102.220243591126,
    "hselphnqlvecmmyx",97.5790352501593,
    "htedybhazfjiueyj",97.5968174680713,
    "htppstzpipwjtuia",113.449925192075,
    "hungxfwbkelospfy",105.206217444816,
    "huoicgalccftwyvz",106.885117073285,
    "hvjwbevmcmjpnknw",110.991565230221,
    "hvziklxqbjbvncjy",106.978432085359,
    "hwldevoubgzgbhgs",99.59900733883,
    "hwsgwbkydspkbben",112.965375082174,
    "hywzsmogbhnfcaxk",103.2908434447,
    "iadmwbxpppukpjyh",107.486814584848,
    "iadwyxxyvkcpyeus",108.033124863782,
    "ibjlpnapcnsmgugu",104.322244914987,
    "iditakunbaxfjcmc",108.58385425431,
    "iemmvtjtejhlteqa",99.2934302751269,
    "ieqgavmmxulqlvvl",106.36407488204,
    "iexbeucevqnjjbcz",116.52376538576,
    "ifalilovsdszxmjm",107.793874212079,
    "ifrzhyqsimoeljaa",96.7085319479887,
    "iigklaveqvybkbid",103.972029625777,
    "ijxmcnthqquddvhc",108.729804673958,
    "iklmkdrwatltidff",101.783874979473,
    "iknapxqudqotqiig",92.5820388706505,
    "iljhlfeengkciosq",97.486305193936,
    "infvsqmvfzjpyfae",106.228485422296,
    "innngarflvbnwntw",105.921493304346,
    "ioqpncqqlflrjzkj",111.175907944086,
    "ipauahivutejsrev",100.006988219644,
    "ipyrvtdugjovdwzv",106.615574886147,
    "iqepotyqjqeebzix",96.7114832545074,
    "ismjlsoibleinjdp",107.718686806649,
    "isyektlfmcpmotpl",103.77725986177,
    "itmcxdqtvddvmanj",104.5691669407,
    "itvlnddnkkmyemme",103.928439443845,
    "iulvirmzdntweaee",87.9755510792354,
    "ivhhwynrahlruefk",94.2175260392604,
    "iwhqpdfuhrsxyqxe",92.6883068363331,
    "iwxvflrheripbuvw",101.860222480484,
    "ixbrfaoerogqomah",97.4131535199764,
    "ixfiagqhmszowdmf",103.407336870623,
    "ixwsqebjjdlxcqsq",93.4639756981393,
    "ixyvsrnksxeiqbve",108.506024496354,
    "iydbustazndekvfq",109.343730141633,
    "jakvzvdollijyhwm",96.4088514973134,
    "jancrvhjhcbxreda",112.889142295469,
    "jbvhqxmbarxynmfk",119.398654641353,
    "jcefoutonncubdss",104.480272248218,
    "jcpjlgfslytgmbjq",102.633750484462,
    "jcxkvyjnzflnlzvh",97.5050145817946,
    "jdsmqjpfexexznya",112.541842246993,
    "jeckddxjsdolnuhe",97.952407237269,
    "jedhlhdmkdprvyex",110.272830800457,
    "jepialiqqsttgcid",97.3113375393053,
    "jghfkxkawqeujuhj",110.087046973573,
    "jgkpiuuctpywtrlh",82.8942466348264,
    "jhafhnhmasllifix",118.271520351357,
    "jhdjdpthkztnjvmb",91.378320701956,
    "jixkbeuswaznqplh",112.048658355552,
    "jiyhnfvmyyrpnzyx",101.951779476152,
    "jiyzqszfywhdfsil",100.562964901004,
    "jjjvjaxpzvlbryfd",95.9076250527038,
    "jjycmklnkdivnypu",102.298302953298,
    "jkarjtlhihuxqzfm",101.187820383787,
    "jkguypwgxebmtnkx",106.338815061321,
    "jkhjcfudwqurdoex",104.597754531549,
    "jkwlqsmedtplrvtj",102.475788418916,
    "jlhzkuikphkxcigk",99.2156425251089,
    "jlibzlturkpyjavf",104.944048727049,
    "jlxizhsfukrheysf",100.896402389464,
    "jmlbcbnedxdoagqm",106.328269508306,
    "jmycebfjwrkqwsxi",102.295403630493,
    "johsjccpkithubii",97.5878762920086,
    "joosvbazdbslkqgx",102.348085246667,
    "jrwdpzrmxqlzzepk",101.680500712497,
    "jrwemlawxsvnwrxv",103.404977621235,
    "jskghzhjrpywrbfn",110.911443155969,
    "jsudrcgsrfddwixw",102.518555764948,
    "jtedxzwqoodxzcaq",106.481946336292,
    "jxlbmlxexeucwbue",113.99816898906,
    "jynbrbzntxrssxzh",118.171074191112,
    "kalfshwbcuoobdwe",95.4369187386342,
    "kbgblyclstrmicux",104.939273945704,
    "kbixxyjwgxmbhcsa",109.800283237547,
    "kbnrpawcssaxrpmb",102.073278899719,
    "kbqauyzezmwspqvv",106.575426622562,
    "kcfhiwouwwfjqtta",102.962264840338,
    "kcjttmlajpvbntkn",99.547952347402,
    "kdsbtuikoaulynsu",105.031937815806,
    "kfurwythfncqbrxs",103.629455112649,
    "kfvusykzaeetiqtt",108.917162735146,
    "kgezpfvpmpmdicts",109.953722906545,
    "kglgveumqmtwrqsf",109.678470014756,
    "kguahfjnmerrbtpp",104.348595645554,
    "khwbllfppvhgkgzc",104.428472750096,
    "khzmqnkqbaqvnakh",109.430868166149,
    "kilbdkfbpczjrqek",106.218200649039,
    "kjdumkaiaeblbxtt",100.381264493988,
    "kjhuznifzeghfdra",104.185510838124,
    "kjogjnoblzpoxgyr",103.426419299191,
    "kkxluqnhrmwkfqnh",102.88437119672,
    "kmlnlefquqpparsa",105.574977854736,
    "knrylcwjpefiqlma",114.962487072699,
    "kowgdytyvjhvcmta",102.084114277416,
    "kpciudedjlrqsfte",95.6968016005918,
    "kpnwdujiylvsiuhp",105.924703778465,
    "kqubvdyyovhfxtpc",104.555617989892,
    "kqxycgbergacgcei",106.841742984503,
    "ktpoqrjuewxmkjqr",106.645977933301,
    "ktrfapbareyzyyyq",116.027578252833,
    "ktytfazsvecrjvzl",104.274313428765,
    "kumhekfclnypkavw",106.198538021685,
    "kvcddisqpkysmvvo",95.896740056797,
    "kwuuuvwdrjkyqyfv",104.737586409427,
    "kwxjejihbgmtnagf",102.599669346474,
    "kxmtwjjyzuqqgmjw",117.253241182973,
    "kzhhwebpekxgvfsl",104.726692625534,
    "kzqcxkrdytalrphb",101.218345411203,
    "kzwthrslljkmbqur",89.4969963278374,
    "kzzakxocsxhkvslf",126.521063304011,
    "lcokgbxbqigkqzcw",111.15833081374,
    "lctcvcvytpesgryp",108.309447336689,
    "ldejndeewhhlcvgc",108.485723513555,
    "ldkzuxzespcgajev",105.654566901232,
    "ldnocwfyeejbmmcy",108.392592482278,
    "ldxjynecsqlswvbq",92.2442102492361,
    "lfzbrhthlxhnmhva",95.7904496656561,
    "lhamctzhosdtmdix",106.473086271239,
    "lhgeydlzsntbaqzj",108.013895945692,
    "lhyhsxrxdftbsavk",107.98163025943,
    "ljwfegchielwaghb",108.475710455438,
    "llkwlxfjdmrqmdgq",101.178397307142,
    "lmqoiaqyftqublmk",99.2664056784204,
    "loomciwexxewgiut",103.850869758595,
    "lpwtmtiwkgbwhufg",108.761996260461,
    "lqalilfrsznnxarm",105.530897811191,
    "lqkdgbosdzrtitgx",101.624626822104,
    "lqohoawdpvdisdiw",105.607596895513,
    "lqqciehjjdtelpwa",100.444928714161,
    "lqsgdewyevczcvwf",105.338186763057,
    "ltdxvujhaocpnmzf",108.929776349757,
    "luwiodhzrjjobjlw",108.871643429916,
    "luxhsezouvtbkbpn",108.376462965696,
    "lvpcmycoagwxqpag",105.738201437845,
    "lwclhevnunilhrmm",110.357150624153,
    "lwhjrctubjkbhzmu",108.005752008655,
    "lwrjcljtxkokvnes",103.622579146308,
    "lwtlsafdbhymtibi",95.1407917632341,
    "lwwzmxipnntydwir",111.667066373647,
    "lxchmlyoaiocynox",106.013417266807,
    "lxhecyqzfsucxgqm",104.757367285282,
    "lxjkslpwiofoynao",105.47771194342,
    "lxvjgyjdszxtcryf",101.515001537341,
    "lybpmhaivmaqtmsq",108.214917193989,
    "lzsfpyidvnkaxnvs",113.305476786388,
    "mbjevmuapzxqjnwg",115.087967225268,
    "mbnozlcufjgvpcdb",108.881413317312,
    "mbytpqiuixyvpaab",101.668334191075,
    "mcadxmmocjhzzbtt",106.57021224369,
    "mcloznejvtelpcan",108.154603058249,
    "mcuawemlwwgaiesn",104.358255326953,
    "mdiqmxwkzvnpeaop",105.687820817391,
    "mdqyvrtwekmeflye",110.661046591094,
    "mdxtphkujabwpjeu",97.7769767054944,
    "meratbpknllwoefn",107.957238113251,
    "mizxbkgdiuoehddq",99.8093974578305,
    "mjpgppxzelxrbcnt",103.946667862882,
    "mjwrreshlbmzkwmc",102.543269834905,
    "mkbpzddzmalsleud",109.743448498954,
    "mmfquhvxcmjcvmhz",112.705920465115,
    "moayoogjmiizcbez",104.736525915321,
    "mpmchhrcazhsvjgc",105.845419974445,
    "mpnamiwsqkvamhfa",101.052584115127,
    "mpwepwxyokmciojj",96.2523740142148,
    "mqsiquclpholncqd",114.850992212916,
    "mqzhmlqqmpafpbqw",106.772515600296,
    "mshhupropfijhilz",115.431210155338,
    "mtcsefxrgtfdqous",101.545528491724,
    "mtubnuteguketfck",113.370103014681,
    "muixzziwtwouzapq",107.860047097344,
    "mxmhlvlmychxzork",111.001707433237,
    "mxytuavlfghapjvu",116.039145376331,
    "myfrksrutuknkcnq",107.343779655303,
    "mymdahqxtsywqpdn",107.108807977137,
    "mzlcdmigakbbuzli",112.891053883855,
    "nbxjozrynlospbso",105.823459346132,
    "ndepxuvlaiqzdnan",116.487622430122,
    "nfmbusxwwqhsaquy",98.5468860682657,
    "nfrqxttuhpuqvwti",111.372757355072,
    "nfxbfvlwvmxfproe",106.597161967585,
    "nggwrmvazdxdjyfh",109.20085306506,
    "ngksfbgkdeufmhfy",103.354483693421,
    "ngombkqqomblyxwv",94.3837014851245,
    "ngpgrthcqiirdsux",109.21638199539,
    "nhembilpmgrfjifn",110.162904440376,
    "nhmkqmpmstaunzqh",104.860986699921,
    "nhoebceeiacnmvym",99.871687932097,
    "nhqkbmwihkfvhjxx",108.410600299772,
    "nhwgapjtnadqqaul",91.7117550558375,
    "nilvygybpajtnxnr",101.731034469521,
    "njcwousmigzpursi",114.458991450991,
    "njujuhbmnqusynwf",110.371102220341,
    "nkktflvfoasvkvht",114.183227639482,
    "nkueyjctyasmotny",99.6568843548059,
    "nmhahirmbvqxhxgg",103.053352496947,
    "nmkzmncfytfwyfvt",106.676979580038,
    "nnzwevftfeodipkn",105.764599876232,
    "nofmcfnaiuzlqgrk",96.7614138825796,
    "nolayrxwnjwzgtoo",104.799085307422,
    "noxmlxlzirrxdriv",108.016255178584,
    "nrmzpcqkbzgmsdeo",106.160744698904,
    "nruhduwvuytxnfvh",100.907256370441,
    "nrwphouoeazzmbmx",114.586819686023,
    "nsgbpbjvswwlhvmm",104.608788773075,
    "nsymgnybdjqxudvj",98.6457982589463,
    "ntjpzidotcatossl",113.402415692366,
    "nwaavqeweeqaryzv",102.529686360744,
    "nwfvqtdnlrvhdbuc",99.60706011067,
    "nxwedpnhirijkodc",102.067943158041,
    "nyrtstlobluggnkw",101.356529359055,
    "nzanewsbtbnpgrom",102.118703635756,
    "nzgfjmknhxdezggp",116.337208967679,
    "nzxlhibmhrtafeav",104.06740910852,
    "obkqpwjualnnwgrt",116.515483947565,
    "obtymepcippfwigb",105.614999746248,
    "obvxygchobqafuzw",100.971187972953,
    "obzgnvzzatnjoryi",114.812686737134,
    "odjkyxbmtxqhkflm",111.530410291629,
    "odpuaztxnyumdvvc",101.206537163678,
    "oeexhaebfkkjfpff",103.046450692063,
    "ofrkezlcbbluncri",103.904379556919,
    "ogdxwqtrpclsxeyw",110.686901143489,
    "ogrmvnhwyeydwcxi",99.6956050606362,
    "ogyvyvhcaefqrlgk",111.110592195139,
    "ohgtowaarzphsifb",101.600244187848,
    "ohxrgpugowiyinhv",104.099199576358,
    "oihtzffwsrwsjnfu",107.315075314028,
    "oijipbtrzkghftpt",108.147105094896,
    "ojribuhtopqgkqpp",103.127098279321,
    "okeuihmplbxhxceo",103.882078203892,
    "okzpgwvslpvgceva",106.924280860961,
    "olupoctwepebdqqo",97.2105996280601,
    "onzjhhtppsfaiacz",98.5904532172771,
    "opojibguvnupidif",101.22496464138,
    "optzzqvbwwriedfo",104.582598390816,
    "oqbjvmfvjonftdxi",107.878090292614,
    "oqkxqgmcsytmcsjz",101.556840791846,
    "oryfrzxilushvigq",103.090054422249,
    "ospbwzzmmxeovscc",104.593045486832,
    "otrziwxmbpndmyaa",102.796393896427,
    "ouhkmefnnchsggpl",113.266200624408,
    "ovhdtvldyrrurawo",111.493391085729,
    "owkgoejsxqlzahbz",110.064992786642,
    "owkiszjuntmwilff",99.1973265779904,
    "owrozlxfshxrcgvh",103.627267106941,
    "owxrlgxbigikfgtm",99.9965919693078,
    "ozmdlzfsareqmkon",97.2575056145302,
    "ozpyjjijxdpztngv",102.743182560493,
    "payritakwxpyzwqq",102.399913029078,
    "pbrroilhklrifbwq",106.380155458705,
    "pbwbzedhenqmpfqt",122.788559457235,
    "pdljbgzzhxrhnqmu",101.327610168531,
    "pebdztssohmloufw",113.852952182494,
    "pfkfojczxwevqesz",113.801618150978,
    "pfvqxmrnkptcrhet",103.480147653149,
    "pfwcfdvpkuyucnkn",103.689211142066,
    "pgkgdfabkhkbviht",112.434426083266,
    "pheduvdlnmrchihf",101.291385310215,
    "phprbhssfhrtbeue",105.53945655824,
    "pijaubxodtxcsqjp",112.078528242539,
    "pjbnwqhnqczouirt",90.0630167022159,
    "pmxjblqhvpwflkwt",106.075709873897,
    "ponwkmeaxagundzq",95.9785278140553,
    "pqpqthiapbycbhor",104.776652328375,
    "prtnwsypyfnshpqx",83.1539068484613,
    "pselomoxubpkknqo",103.403907420405,
    "ptaxsjwbissrpvdm",100.740506796405,
    "ptbudvgjgycmmsdq",105.586770824254,
    "pticuqiimwdrkpdy",108.659997238491,
    "pustczakchcimwuy",94.7681957503494,
    "pvrjjyumueakzstw",104.530671548594,
    "pvyfdiggxtjoyhqf",108.512011987818,
    "pyhcuhumhsoodqwl",119.530427775399,
    "pyholyswkkqjmxlj",100.317625725465,
    "pyykjiriqrhjduly",94.2869313517974,
    "qachmbxcslsazphb",106.222169910802,
    "qahlidfcpdaofkwm",106.675179276879,
    "qbkipjmisqllqwzy",106.479468814512,
    "qbohnomeacnwdafj",107.833622128775,
    "qbztetcodwhfmoyg",100.88588031194,
    "qcykqtxlqnbcqfct",103.867640411933,
    "qdhfzxrzisivuhbx",107.78435643906,
    "qdmbicmyqrqalixj",105.947474910787,
    "qdsjznqzjxlekjtp",97.7978272907952,
    "qdvjpkftaveygusd",99.1315345004323,
    "qewzxgvvhqhkfcxe",105.760059715235,
    "qfvolfbvalczrcko",101.782283927029,
    "qghhvatpvekejzpf",101.350089196734,
    "qgnqfinpenszbzig",110.412427441034,
    "qidpxyunryowizua",110.967311168073,
    "qjkwsppqbsgsvjwa",109.70142417361,
    "qmahqrjhkxvkwboe",108.792374046387,
    "qnesuhpxsptzihzg",101.295550762887,
    "qnixeczkijjyiprb",100.983303947227,
    "qoflnrycwjlbfmow",102.866605507108,
    "qozlaoxmwusgalpz",114.792937667269,
    "qpcebxmotqhildhx",99.7951378292998,
    "qpjdblaqrqyuoaqk",104.941596803109,
    "qppmxxfbqiiallmp",112.553733656667,
    "qqmkwgdqaimwcbxo",105.016188444932,
    "qrgsdbjbjwwgirvo",100.793841357448,
    "qukbrubjquwstnyf",111.705046423703,
    "quslbttvcitxzeiy",98.7610974074197,
    "qvwenzdmnwecdiql",95.5953629870086,
    "qwcrrrebwyeauczj",103.780564884236,
    "qwedbcvlquqfoycc",115.149930284288,
    "qwqwzvbefvgugtzi",117.765729108298,
    "qwshkzmlvlerxsov",109.671147517088,
    "qxksnnsrnebfkwqs",98.2244686839772,
    "qxnyigoiwisibpko",115.926774509757,
    "qxtqrwxfvuenelml",104.288685755176,
    "qyqvfzuwfpyztbla",99.2622085810568,
    "qzgaezfhutbcnkuf",104.6940810776,
    "qzkbvcycbyxrgbqk",106.51120193189,
    "qzrkqxhgbqfyswsj",106.777224796882,
    "rabwrzdzwjjdhbmx",101.110420421907,
    "rbxibrjokiihgfjb",104.889109107092,
    "rclsneerlfasdcpi",100.443269441094,
    "rcxmbwwsxkkkyyjs",99.4966743037275,
    "reolzfmikorzxstf",109.465647810911,
    "rgfytoxurocumuxu",111.996133161602,
    "rgrpzewhrznrqrna",103.649156412674,
    "rguedwefqmzdxowu",103.112170637388,
    "rhxboadaoyvvgflk",103.603919490174,
    "rjhfsrwtoqfqvuqu",102.099252325408,
    "rlkrrmxxdgaxangi",101.206243112999,
    "rnrkbyojyiepdvqv",103.429963466846,
    "rqaprgqcktgrlxnv",96.5531492354781,
    "rqklbykswxeuovdn",103.76335538177,
    "rrlvhbnzrdtphqnl",102.760997193286,
    "rrqbtdjvuwwxtusj",88.7667063210898,
    "rrsrcesavzhbjqwk",108.952648467512,
    "rsphcdnwdddxhdvb",108.650396710625,
    "rthsjeyjgdlmkygk",84.3293381377973,
    "rtqyfobkpliuutfx",112.958251106125,
    "rulqevsymrlwrsrz",97.7076212034293,
    "ruposftqgswlcyou",103.473405802366,
    "ruyuflpnypnsgkbq",108.446363420371,
    "rwrevaiebpmviwqz",95.1983383290162,
    "rwtwnvhjqabvovnz",101.165399308982,
    "rxyndewyvbophaku",100.562316317662,
    "ryjiidsxttvdcpwu",109.288435230603,
    "rytmtyltypttvqjs",110.722588117837,
    "rzjssfxzzoddvgdc",110.229459892613,
    "saempmkfulqhwfqk",99.8838961024171,
    "sboaeuuuhpsjujpz",105.682498969763,
    "sbrarddcurfhmmqk",102.811225458981,
    "sdottmimvqvfhzlk",102.04097289673,
    "sdvssyrvwfwmdccl",101.748822117207,
    "selnccftdsqbiurb",110.998726648212,
    "sfcciovhmwqehacv",98.9975909458247,
    "sghyfposeljrkedw",107.275344991777,
    "sgknghheolfpzuid",103.145908005865,
    "sguprofjftozaujc",103.463895630587,
    "shemwbbeliuvnvvm",97.7434462872089,
    "sioekxjbocpzrjzi",99.6028515726734,
    "skgvahbwdkddoxha",96.7067278395813,
    "skwelgffvlzgmbro",107.867287713742,
    "smcawzwicovvejgm",104.013322673909,
    "smynsodmtrrubpqq",104.524266670824,
    "sncpkctrqcditirm",113.000814725941,
    "snpaaoiipfuxmvol",104.478108149927,
    "snsnxmucuccvqfvz",100.234888410359,
    "spbjeokdemicpdey",100.814228759162,
    "spqqpwucqcaspwkb",112.270281446868,
    "sqqvhmadjqegpsps",116.088095323655,
    "squxtuwvjnzbhzsc",106.764384228087,
    "srgqbkjrwdbikmzq",101.434686028792,
    "ssnqyyteovyaxylf",100.84208611303,
    "stgeqvsewqntykyo",101.112833613524,
    "suajnmrxuunoyngf",104.450923490544,
    "sutdaojcvfqmjnwg",96.2358833673458,
    "suycgjdrxxvxgmha",106.65408066073,
    "svcvmlpsqtzbrmnz",105.76908516545,
    "svmjzfcsvgxiwwjt",92.4703995292541,
    "swacqepcxnosmcll",109.543261720805,
    "swjkmyqytzxjwgag",102.313692920943,
    "swxgkelaxkoffszz",105.600562576282,
    "sxmsrnbwrnvfjcvp",109.164014602052,
    "synvsxhrexuyxpre",104.100300359905,
    "szduoosmrfqduakm",109.875461604171,
    "szlkmablxrjoubla",108.754142614218,
    "szluwlsqbkcnchxg",108.045128761162,
    "szvfwsizhxrbklhz",100.922055432988,
    "tafluhgrtixdlhpv",96.111348525684,
    "taovawittfogygzi",101.773708852944,
    "tbpblaaxsajjlyok",114.929767391616,
    "tceovgpqjjopitor",99.0420319727137,
    "tcnfpudadgannoey",104.179064705781,
    "tcyceqtrfusfmkpy",114.564397854549,
    "tcyogsbbufjzekla",111.707835792251,
    "tddtoayhfpdtxokp",102.612118471247,
    "tdgkjlphosocwbgu",100.560143681506,
    "tdldeeccsirqwpcj",105.143732022813,
    "tdozuksvtvtqcykp",100.350515916939,
    "tduddcyerrjazjsh",100.155385092455,
    "tdvzvrkldmrkqeth",107.72532163705,
    "tegzsblugaczvdmy",115.932558250136,
    "tgfhgapnsxiewemd",105.846612627008,
    "tgvulwtrjyegawlr",96.3889554065932,
    "timtcrwibllgvgxy",104.206059396412,
    "tjdlkefrbysjheap",98.2625186566732,
    "tjvewbsfsiqtqttp",101.722300437634,
    "tkomxtfmozdiflzf",109.487225852636,
    "tkqxtjbbrzagooya",102.333704600399,
    "tksyxmdgogmokuxv",106.640737592334,
    "tlhipnhcbdhvhgyw",107.034820785256,
    "tlrnhgwgduswslyd",112.777783980922,
    "tlspgqlrhuzholye",114.341418177473,
    "tmikjfqekaorgssv",107.501990236991,
    "tneakanblaxyevhf",106.479514179841,
    "toqaaqswchaiyhsk",99.1461411403263,
    "tpzzxliudfwqpopv",107.243269842362,
    "trcsvrxdekscyvyq",102.922140156937,
    "trwedbipujnvnhpr",112.044576373333,
    "tsfyxgkwdidzgzpg",111.46054036823,
    "tsufyplsxyqgndsw",106.157527197841,
    "tsvmlrkmftqbjvub",105.259790982928,
    "ttcbkakfxnfsllyq",107.018579969333,
    "ttwsxzxrhwuzystf",101.134604159276,
    "tupmlwnkgjcgcmuv",102.483750882296,
    "tuudpbartgtwkoms",113.158770067119,
    "twhtulgwsricneea",101.877913609094,
    "txcfordzmkkiicwu",119.596907098432,
    "txgvnaysouvjtkrb",111.152100207603,
    "tyktnjdtbrucursh",103.416034320733,
    "tzsnmmekuhggblhv",102.978328628942,
    "tzyaldqhudfiajin",114.953579091666,
    "uahuglbjdtacoqjt",100.201198256889,
    "uamadghmregezetz",101.17144181494,
    "uaweumzfugnsyqmx",113.279397948359,
    "ubixvbvqnnjdksca",104.453377937553,
    "ubmafwgdsmbkfmwe",108.578941223527,
    "ubttjiaeeuwzcclq",92.987523663608,
    "ucggoqoneixjlxxy",99.9006013459978,
    "uclvvrkbezlvaulu",101.908276431193,
    "uczhoudymbvnhter",108.912187624434,
    "udlfdefgndowttah",103.697827953839,
    "ueujqvpwszzhovbj",104.512827807504,
    "uevoqlbbbmmhkdbi",111.22435285778,
    "ufpwnqycocwwbgqi",104.202471663367,
    "ufqiiuoxasjwxqbf",101.289289752357,
    "ufxhmwifrakfhfmb",106.255016890132,
    "ughzfvxgeziewvdi",107.63700607177,
    "ugwrafmvdavbsrzl",103.59872847432,
    "uhhrakuolcaxvsbh",110.257987460531,
    "uhkhghxuorryhlis",97.6925863451043,
    "uhnrmazhzouxvsqd",111.954084054402,
    "uhqwwluaswbuqqjc",105.096586345194,
    "ujfqhxynqnqeldes",119.743716143816,
    "ujgldnoigollndkj",104.281140443009,
    "ujwlwswdwvbpacnf",99.7075852085377,
    "ukqbsscpgbfatdhs",103.00451836701,
    "ukztcjpqrpetqrnx",105.738227454635,
    "ulrbzlswbgzvmpas",111.849918852691,
    "unlqqlfvajjczyks",109.533229906935,
    "unmxysjyilftwsvy",105.037036758695,
    "uotpflqyvprslxjc",97.1958107781199,
    "uouhydwldpcdzuoj",103.687075823518,
    "updzeguaxbccwpoe",109.297119102599,
    "urbnjdherequimyo",100.054044708044,
    "ureljkoqqvqbpdvx",109.339467409954,
    "urullqqbaabxllxl",104.610685853372,
    "usjukvawgoqplrph",98.6689182703039,
    "uskrsfpueljrtxkg",110.685030524326,
    "usmpkujeknoxdqrc",100.861029512852,
    "utvgeykupnwzepks",104.830660919519,
    "uudtrowerqhfztjo",112.597792245957,
    "uudulyvocjutaxtj",102.537177068062,
    "uureltetaotxxdji",102.945155753814,
    "uwqsodousnydlsud",102.609814836112,
    "uwwrbkmjbjyxutfq",98.4167025959296,
    "uxfxpraspeoqtmbg",105.790027306478,
    "uxpkyjrybttfrluy",108.137575238386,
    "uysllzwmzcsweunu",106.487097104271,
    "uzfgpmnazksmudrw",109.482179865661,
    "uzmldekmvczimsrj",95.0892759736567,
    "uzqtsirvtxcfqnbp",103.877066798834,
    "uzrhbfduaqijosql",100.013944155834,
    "vafxeruawxlvrttn",100.975489855464,
    "vbjxuynqachujrmt",104.836357604314,
    "vbnagxwgmwirhnjt",98.9734789016691,
    "vdetuihriafhetdl",108.311707598383,
    "vdwfxzfzbybhsmay",110.743040524829,
    "vfprhybczhnkefdf",110.586712380123,
    "vgaanttvdscmqmjr",104.197675542196,
    "vikmjrynreazqubj",98.1382199049592,
    "vjarzxzevsdnftcl",106.951757439313,
    "vjxfnjqgvugwjhia",105.263765592097,
    "vnefzhazthgsjuax",103.390171841678,
    "vnsxpowqjjomnmac",113.597147944011,
    "vnxfpyxuciadydrl",101.029244035527,
    "vokpwtikxckeemdi",111.787602759585,
    "vomuvsgbhqzjwhgb",107.073739048675,
    "vpvsuuudxglarezp",102.7912149014,
    "vpvtqlxqaiejzrqo",119.238199916578,
    "vpzzxdehhwlzsgrp",94.8461779024949,
    "vqdlslwzvwucentl",107.104751287795,
    "vrubhbzjguaxfmlc",105.459187633277,
    "vtgypuzvlawmkolb",115.529644905393,
    "vtmylwmvnssatjlh",104.425934468232,
    "vudjowytbogxkrcy",97.2836617899372,
    "vuqhkgrnmheydqku",116.083415713718,
    "vuvbtxegkdifkviv",109.452131671893,
    "vwerelrvyumnkbwk",109.160608982801,
    "vwjjtxhpebfhzzck",112.611182274847,
    "vxqfscklywhurrjp",105.515145388254,
    "vxumxjoeywcphfoo",101.439749243786,
    "vxvmxuncsxygbrzd",112.947516435408,
    "vylopbnfewdzeury",108.876712062066,
    "vzinjyeuiebqjmep",116.788621686804,
    "wchxrbrhstsmhdsk",95.7069776058246,
    "wcjrtzzkemciejsz",110.688568466078,
    "wcyahxrmwqvhmadq",99.7445511222354,
    "wehkqzwvbeonajcu",100.509219620471,
    "wffzvseexopfwwjy",101.03525039359,
    "wgruytvmfzalzrtb",101.780436578137,
    "whcqrtwarljaqocm",101.024821541559,
    "whfzavgfbojmgezm",107.074482794725,
    "wiyyghhcxezudyxg",107.201472830019,
    "wkzwidzltxinpgen",111.88849820981,
    "wlttiymytfacsrli",119.341219605831,
    "wmmjbitjeevklkzj",103.993999912798,
    "wniogbpezwqrinyt",116.456119267589,
    "wnrwtzbbxbvnmbqd",115.637621278334,
    "wofhkqytrnqvbije",121.067082683189,
    "wotupxkyxwcienzd",103.836895820983,
    "wpsyqubfrhdspxkx",97.2014311577688,
    "wqqxbrsnrtnuxjjl",97.5009107913392,
    "wqsapwecaqwzorqn",108.601143148048,
    "wrpeamdcqpawnqag",106.832139375569,
    "wrvfnbtdqgpsnzic",105.461636494202,
    "wrzuftzqwoiwsmfc",101.457370029001,
    "wsfgxnwhxftjhpxw",102.045337095318,
    "wtxtgodhmneofvzz",104.018175685188,
    "wvnsmznngunxhcsb",108.361082939513,
    "wwmsynqlijbriqxy",114.563477409531,
    "wxcohilpavlwlnze",113.824959800058,
    "wxzfbqtarfurwcfw",100.244189344092,
    "wycspyzpbmhbnmda",108.238037898738,
    "wykcypxicfqltavz",107.320879314752,
    "wylaluxuyuqkytus",115.894658144278,
    "wyopladghryqlrlb",103.776970388967,
    "wyoxchtoecahbyjm",100.853241917524,
    "wyqgeeclrqbihfpk",90.2695091148853,
    "xaaujpnniyiuhfql",108.382953378408,
    "xabkvgvnbqzrmnyc",95.8237409364514,
    "xaklvfxsplowrglp",98.269284003105,
    "xbdjgmfolqdfvftr",109.924168953662,
    "xbhnetrbyfixuzmj",95.7032875821511,
    "xcpcxksgiefkqznu",97.3623119879921,
    "xcvdausiwfrjukgn",101.910588223126,
    "xdeuhuabvdhjipnp",104.992622100825,
    "xebooruxiuwbpzdc",103.523276315482,
    "xewlloxrajhpbuwy",116.458447956941,
    "xgikerzyofvqsmnt",101.278378161144,
    "xgvsuftfggoojbdp",102.279325450722,
    "xhczitnzxmxxebeq",103.975673883837,
    "xieosvyuphbcyzul",108.321787133331,
    "xiesnbkcyzrpzlyq",105.534993269378,
    "xiwnirovwicymtif",103.817312583228,
    "xjaddkudsebowzen",91.1621981532252,
    "xjnjmxyqemqqiejp",99.3909215083445,
    "xjsnlswprucbsehn",98.5142915584777,
    "xjsozzwcppavldee",101.562480583688,
    "xkqlmdeookbxzzhv",99.8943082653307,
    "xkzehzohmfrsmolg",98.7319240158811,
    "xndokrsndaodfknp",102.688799396812,
    "xnwekpoxnvwckfcp",118.626226775668,
    "xoetbcowyxwukawr",103.952161941776,
    "xokomvoaaiyuedhu",111.548504061977,
    "xoueplwxwxrzasti",106.723980936579,
    "xpxsjmglcvcsxwdy",96.9445095423784,
    "xqascjfdlnlxubce",107.7518177128,
    "xqydufxhniyjunrl",103.12183919087,
    "xrmvlihddtxlbzvw",113.497164847374,
    "xrxxucncrqtcgixl",103.105394446301,
    "xsfzlqjhqbcpswcz",114.283357184953,
    "xsmqbeukcqahbfgl",104.232956784397,
    "xsonelzsqbpcodxe",104.131914778671,
    "xuioboiuzasnmuva",100.331182853073,
    "xuxdvcabcanlgmst",107.989408730143,
    "xxppvuhwmnezefxy",104.611381853712,
    "xyuwuxlpirkzkqdb",94.7736636519605,
    "xzdakcdqrnwhtpdb",108.729537705465,
    "xzdsapxqliboezbc",99.9572061357747,
    "xzwnliotgalpusga",106.122921486874,
    "ybbdbpunwekygnto",100.271486505889,
    "ybmkbrazyartpatx",99.8190601523342,
    "ybsenzutfrjternf",104.924312065814,
    "yfalryaixpzfoihd",107.488149010364,
    "ygctwjsvugjhuylz",109.59032999536,
    "yglorajvvrsviget",105.518957444865,
    "ygqjgwzgeierkcpj",99.2434804736694,
    "ygsnfanduarpqvrn",98.8011399170256,
    "yhcelpjnbxpsmoez",102.689550093862,
    "yhotwssicqxqetep",113.983608231708,
    "yhzuhoixugafkonm",111.071998840754,
    "ykzumnthkadrzjdb",111.998360901774,
    "ylboorftnzombypn",99.7453400922225,
    "yljpdxzmshdpmyhl",102.708360685464,
    "yobesdjweimafxnq",116.7031083938,
    "yohngmkmrueenrvs",107.281324435764,
    "ypgcfauffeqpeerz",104.56409457834,
    "yqrgxthbbzmruvwy",103.524530259167,
    "yqztzexmqeyeirmv",117.035542524369,
    "yrfmcopbrlmfinuq",104.930803358684,
    "yrnbnvvghdwvpayv",116.030426850891,
    "ysmwdnymkzsgskpv",105.858528064557,
    "ysonsqntnqnqagnn",101.168328021442,
    "ytckjafhdfppmrhv",113.254105543077,
    "ytornvfpgbsoizqr",109.881603057189,
    "yttvzqeuddvehiqu",103.065048007647,
    "yvlkrzgjhwrlyihc",100.970490054492,
    "yvuoearellwavkzs",113.405924830848,
    "yweystgylcxxranw",96.8527697879975,
    "ywnimisaozuyjomi",102.73295504849,
    "yxovdmyzjzoutcek",106.827006370813,
    "yxzgvihpyqafgdmy",103.098390234875,
    "yybjogamsfqljfpu",101.975804748263,
    "yzdbjmwwtofxmpaz",101.453557503037,
    "yzxgnwgpnrdprtbh",96.3329568328123,
    "zagpnfpbwgeyeufr",96.1334632881829,
    "zaiwfnldcznzgrfe",108.948586428717,
    "zakviitdfvxsgkow",106.883116143781,
    "zayyxxzvekzuooyq",103.175656566809,
    "zbprczwzmlxgqykc",106.867853129117,
    "zbqqfjqpoluazvlo",108.477569635265,
    "zcqptmhakcmihiry",101.50427562261,
    "zczpxdcxdciitjcu",103.070793694695,
    "zedpyrfkmhzqxmaz",98.0045088633076,
    "zeqmrfiqiusygxdd",104.374893053877,
    "zfylmujpvzgqqfxo",101.944140609837,
    "zghfsejpgrfrqfdp",109.406271587129,
    "zhzmkwjsrgvudmmw",107.912967854143,
    "zkreetxvsoihwkgo",94.3201130833818,
    "znppxsjvuytcambw",103.694761598938,
    "zojbyremtnxajomo",105.540276856863,
    "zoypfizhpbtpjwpv",85.5652252893566,
    "zqjvnptshpgofkqc",116.834800290422,
    "zqruwnlzuefcpqjm",85.801176248403,
    "zqswdfwtkyehitft",103.549127710972,
    "zrclcvscjwdbabii",115.614614371351,
    "zrfduayrhhofpqtt",105.699195842602,
    "zspzyfdefowgwddf",109.596730924923,
    "zsuewlbquazyrgvl",103.531063966657,
    "ztwsplndgicacmuu",103.885200919079,
    "zukemjnabmcizdzn",105.142689225295,
    "zuqrdemwihnexkpw",109.372076530488,
    "zvdpnzgvkjkoophv",101.607573034033,
    "zwutaiivgrxnrwat",113.14399144417,
    "zxvcbwcwoqnkxxbs",97.5409989954881,
    "zycfmwxhaaaxdwpb",108.998514214692,
    "zydfvjqmmwhyfuyy",99.4223410781483,
    "zzjxvhegwmgqodzk",105.881982704136,
    "zzlzzujtugbfpsvv",102.307345335502,
    "zzubfikjmmfsxhbn",111.253465709503
    
  )
  return(vhmm_mapping)
}
  vhmm_mapping <- vh_groups()

add_vh_groups <- function(input){
  
  input <- left_join(input,vhmm_mapping, by = "vh_make_model")

  # fix for new vh that aren't in the group list
  
  input <- input %>% mutate(vh_make_model_enc = case_when(is.na(vh_make_model_enc) ~mean(vh_make_model_enc, na.rm=TRUE),
                                                                TRUE ~as.numeric(vh_make_model_enc)
  )
  )

  #input <- input[ , !(names(input) %in% "vh_make_model")]
  
  return(input)
}

vh_order_train <- function(){
  
  vhmm_ordering_train <- data.frame(
      stringsAsFactors = FALSE,
      vh_make_model = c("eokuiduvnrtzavmr",
                        "lqqciehjjdtelpwa",
                        "synvsxhrexuyxpre",
                        "drptidaltxzxopwv",
                        "qxtqrwxfvuenelml",
                        "swjkmyqytzxjwgag",
                        "yvlkrzgjhwrlyihc",
                        "ptaxsjwbissrpvdm",
                        "lqkdgbosdzrtitgx",
                        "ujgldnoigollndkj",
                        "mdxtphkujabwpjeu",
                        "rrlvhbnzrdtphqnl",
                        "ofrkezlcbbluncri",
                        "hselphnqlvecmmyx",
                        "yfalryaixpzfoihd",
                        "cmmuslxsfluvfyof",
                        "cxxzogxxkmkjwqui",
                        "xdeuhuabvdhjipnp",
                        "quslbttvcitxzeiy",
                        "tlhipnhcbdhvhgyw",
                        "joosvbazdbslkqgx",
                        "hcfpedolsygjsofb",
                        "obvxygchobqafuzw",
                        "jjycmklnkdivnypu",
                        "aewtdnpoiopumymt",
                        "huoicgalccftwyvz",
                        "cgrdxjyaxssrszjz",
                        "nsymgnybdjqxudvj",
                        "kbgblyclstrmicux",
                        "dweqmfoluivgiayj",
                        "dyzvyrmcdyybbddd",
                        "gjchrdhbeixppooh",
                        "qewzxgvvhqhkfcxe",
                        "zspzyfdefowgwddf",
                        "gqfadgvnztixxbmv",
                        "zakviitdfvxsgkow",
                        "xjsozzwcppavldee",
                        "xxppvuhwmnezefxy",
                        "dlrodwgixwmoquny",
                        "asmpttrlkodaejic",
                        "wxzfbqtarfurwcfw",
                        "nilvygybpajtnxnr",
                        "mcadxmmocjhzzbtt",
                        "xkzehzohmfrsmolg",
                        "urullqqbaabxllxl",
                        "byvoguptigfevpyy",
                        "guiimarisyyjqnfg",
                        "jjjvjaxpzvlbryfd",
                        "ehapkksqqcbofeid",
                        "gfhjqtkgvomiygvx",
                        "usmpkujeknoxdqrc",
                        "dgwbxitzfzbegnoc",
                        "eudwptcohxaazhpt",
                        "ukqbsscpgbfatdhs",
                        "dlrvgwmumnwcjixm",
                        "tdgkjlphosocwbgu",
                        "ctachoeiozcpkmst",
                        "onzjhhtppsfaiacz",
                        "gguphuccgeqyojbl",
                        "pselomoxubpkknqo",
                        "iwhqpdfuhrsxyqxe",
                        "ammbrasbxojlitmt",
                        "vikmjrynreazqubj",
                        "cazrxylvhylncoze",
                        "wehkqzwvbeonajcu",
                        "ajtardhciglimsdi",
                        "snsnxmucuccvqfvz",
                        "gwptulznqgygeegy",
                        "ieqgavmmxulqlvvl",
                        "vjxfnjqgvugwjhia",
                        "ubttjiaeeuwzcclq",
                        "rxyndewyvbophaku",
                        "alrfnehgsdtsunhm",
                        "svmjzfcsvgxiwwjt",
                        "xaklvfxsplowrglp",
                        "wyoxchtoecahbyjm",
                        "pdljbgzzhxrhnqmu",
                        "zkreetxvsoihwkgo",
                        "arfzuuojdtlgxehv",
                        "tkqxtjbbrzagooya",
                        "qfvolfbvalczrcko",
                        "kqxycgbergacgcei",
                        "cqewccykrcmvawlo",
                        "jrwdpzrmxqlzzepk",
                        "wsfgxnwhxftjhpxw",
                        "ewkcexkqpsyfnugi",
                        "rcxmbwwsxkkkyyjs",
                        "blcuqlgntjavsyhs",
                        "snpaaoiipfuxmvol",
                        "oryfrzxilushvigq",
                        "giyhzprslgbwsaeu",
                        "uureltetaotxxdji",
                        "kfurwythfncqbrxs",
                        "mizxbkgdiuoehddq",
                        "wgruytvmfzalzrtb",
                        "xgvsuftfggoojbdp",
                        "sguprofjftozaujc",
                        "xjnjmxyqemqqiejp",
                        "hhrmdevbfqiebnum",
                        "nwaavqeweeqaryzv",
                        "cqdmtwkacajclcml",
                        "rabwrzdzwjjdhbmx",
                        "llkwlxfjdmrqmdgq",
                        "innngarflvbnwntw",
                        "yhcelpjnbxpsmoez",
                        "ettwalwfkzvwdasa",
                        "bsiyfrkwdyptmwji",
                        "arcsdpohuzvikyaw",
                        "ixbrfaoerogqomah",
                        "pheduvdlnmrchihf",
                        "gjblfwqtnckjletn",
                        "hkazsxqvbtmawovu",
                        "qnixeczkijjyiprb",
                        "ebdcmhmtqnfkaalo",
                        "rguedwefqmzdxowu",
                        "zydfvjqmmwhyfuyy",
                        "qzgaezfhutbcnkuf",
                        "tdozuksvtvtqcykp",
                        "zbprczwzmlxgqykc",
                        "otrziwxmbpndmyaa",
                        "djyptluftbfkxtjd",
                        "jmycebfjwrkqwsxi",
                        "nbxjozrynlospbso",
                        "aifsqdniwqmcuqpv",
                        "smynsodmtrrubpqq",
                        "gjpgirzuabhfpkjd",
                        "aceqpjprqgzhffuw",
                        "toqaaqswchaiyhsk",
                        "csxjshhnfbtgjcgm",
                        "zcqptmhakcmihiry",
                        "kalfshwbcuoobdwe",
                        "cufklbvsirnawzmv",
                        "odpuaztxnyumdvvc",
                        "sdottmimvqvfhzlk",
                        "doohwubeqhbkevhr",
                        "fuddhlszptfmosir",
                        "ciuxczxwhwbxdkdf",
                        "ejlwzigdhipvpndt",
                        "ybmkbrazyartpatx",
                        "nkueyjctyasmotny",
                        "bnvgzfegimthyhyo",
                        "kjdumkaiaeblbxtt",
                        "uhqwwluaswbuqqjc",
                        "vdwfxzfzbybhsmay",
                        "nolayrxwnjwzgtoo",
                        "ueujqvpwszzhovbj",
                        "abcepdrvvynjsufa",
                        "tddtoayhfpdtxokp",
                        "vdetuihriafhetdl",
                        "zojbyremtnxajomo",
                        "fuwhdjmdexrstmmo",
                        "ponwkmeaxagundzq",
                        "wtxtgodhmneofvzz",
                        "xiwnirovwicymtif",
                        "tduddcyerrjazjsh",
                        "uahuglbjdtacoqjt",
                        "meratbpknllwoefn",
                        "qbkipjmisqllqwzy",
                        "dpklliwcxycpfriu",
                        "efiskxgaocgqqjvr",
                        "iigklaveqvybkbid",
                        "ugwrafmvdavbsrzl",
                        "tyktnjdtbrucursh",
                        "nrmzpcqkbzgmsdeo",
                        "kpciudedjlrqsfte",
                        "nmhahirmbvqxhxgg",
                        "dsqmtbudvjtnnjwq",
                        "rqklbykswxeuovdn",
                        "nruhduwvuytxnfvh",
                        "kowgdytyvjhvcmta",
                        "rgrpzewhrznrqrna",
                        "cuxaapvakeemmbaa",
                        "jiyzqszfywhdfsil",
                        "anqyvxqouldudiww",
                        "xsmqbeukcqahbfgl",
                        "clsrzyechukbaeat",
                        "eyrwkwxecpzxzscp",
                        "abacekzzrkhtgpcp",
                        "qpcebxmotqhildhx",
                        "yttvzqeuddvehiqu",
                        "xhczitnzxmxxebeq",
                        "gdaxhrlhuilhiijt",
                        "dwhlbcevejvegsob",
                        "xzdsapxqliboezbc",
                        "kguahfjnmerrbtpp",
                        "qvwenzdmnwecdiql",
                        "ngksfbgkdeufmhfy",
                        "vgaanttvdscmqmjr",
                        "agowcnternxraavr",
                        "celpzeaubkxaxxbx",
                        "fgxxyxcbjkodwcln",
                        "nzxlhibmhrtafeav",
                        "spbjeokdemicpdey",
                        "dqqtizjjhjmqdqqb",
                        "hcoxxbfccserxklx",
                        "uxfxpraspeoqtmbg",
                        "tceovgpqjjopitor",
                        "xuioboiuzasnmuva",
                        "iljhlfeengkciosq",
                        "mpmchhrcazhsvjgc",
                        "uotpflqyvprslxjc",
                        "xsonelzsqbpcodxe",
                        "aysnuezuqgjioyyf",
                        "rnrkbyojyiepdvqv",
                        "jlxizhsfukrheysf",
                        "mbytpqiuixyvpaab",
                        "jeckddxjsdolnuhe",
                        "aivacsqryguqpdib",
                        "jkguypwgxebmtnkx",
                        "suajnmrxuunoyngf",
                        "edlxghhjgpmvhabz",
                        "wyopladghryqlrlb",
                        "uhkhghxuorryhlis",
                        "vnxfpyxuciadydrl",
                        "jrwemlawxsvnwrxv",
                        "myfrksrutuknkcnq",
                        "wchxrbrhstsmhdsk",
                        "kqubvdyyovhfxtpc",
                        "aywlgifrijfokyzu",
                        "ygsnfanduarpqvrn",
                        "sutdaojcvfqmjnwg",
                        "gsbeyysssgzgkkuo",
                        "noxmlxlzirrxdriv",
                        "srgqbkjrwdbikmzq",
                        "kjhuznifzeghfdra",
                        "aawqanlavsjfqrne",
                        "ygqjgwzgeierkcpj",
                        "dgwtezteqyzzylho",
                        "olupoctwepebdqqo",
                        "ojribuhtopqgkqpp",
                        "cnicorpxweynumqk",
                        "cnlvybtdupkcwczn",
                        "dqgtaigmpivatpeu",
                        "jsudrcgsrfddwixw",
                        "qdvjpkftaveygusd",
                        "xgikerzyofvqsmnt",
                        "ztwsplndgicacmuu",
                        "yqrgxthbbzmruvwy",
                        "ozmdlzfsareqmkon",
                        "unmxysjyilftwsvy",
                        "dzbwjjmruyqxyvms",
                        "lcokgbxbqigkqzcw",
                        "unlqqlfvajjczyks",
                        "ktrfapbareyzyyyq",
                        "zycfmwxhaaaxdwpb",
                        "azxtekfvyycfmnpt",
                        "ezaffjpqpacrufvd",
                        "gtvhxebtkefavzhg",
                        "jixkbeuswaznqplh",
                        "kdsbtuikoaulynsu",
                        "ldejndeewhhlcvgc",
                        "mbjevmuapzxqjnwg",
                        "obkqpwjualnnwgrt",
                        "reolzfmikorzxstf",
                        "txgvnaysouvjtkrb",
                        "vpvtqlxqaiejzrqo",
                        "zzubfikjmmfsxhbn",
                        "ruposftqgswlcyou",
                        "ablxjgbyowxrfxed",
                        "ajktbllxjzfdtwpy",
                        "btjxvrgfduskmpts",
                        "dohuwjuguzyvqaqg",
                        "dvshwarqhxfcgwfd",
                        "dxpafctvukcmaqao",
                        "eyaqhofitsegmcwi",
                        "fkltkgzmjnzqzlqv",
                        "gdultxlilvdnuwso",
                        "ggzcspiycgszcunf",
                        "hayciibjzwapccnb",
                        "ifalilovsdszxmjm",
                        "luwiodhzrjjobjlw",
                        "mkbpzddzmalsleud",
                        "okeuihmplbxhxceo",
                        "owkiszjuntmwilff",
                        "ozpyjjijxdpztngv",
                        "pmxjblqhvpwflkwt",
                        "qidpxyunryowizua",
                        "qnesuhpxsptzihzg",
                        "rrsrcesavzhbjqwk",
                        "ulrbzlswbgzvmpas",
                        "xewlloxrajhpbuwy",
                        "jedhlhdmkdprvyex",
                        "qgnqfinpenszbzig",
                        "adhoqfsfdpetomvs",
                        "eivjhovgfnfctgjy",
                        "dtdrfrtruyhvbztx",
                        "gdtzpvajphaxanpi",
                        "iditakunbaxfjcmc",
                        "ismjlsoibleinjdp",
                        "jlibzlturkpyjavf",
                        "kgezpfvpmpmdicts",
                        "lpwtmtiwkgbwhufg",
                        "rhxboadaoyvvgflk",
                        "vylopbnfewdzeury",
                        "wcjrtzzkemciejsz",
                        "fdbwfjqkichwdebq",
                        "szlkmablxrjoubla",
                        "nnzwevftfeodipkn",
                        "dcjzdpoxqvgnjpmi",
                        "cictcfpmfdmknnye",
                        "dlbnpwopifytzerl",
                        "heicadwqfavetjwx",
                        "htppstzpipwjtuia",
                        "mshhupropfijhilz",
                        "pbrroilhklrifbwq",
                        "nhqkbmwihkfvhjxx",
                        "gbkevbmczkqhkmoc",
                        "ioqpncqqlflrjzkj",
                        "lxjkslpwiofoynao",
                        "vxvmxuncsxygbrzd",
                        "ybsenzutfrjternf",
                        "rbxibrjokiihgfjb",
                        "ayeiibefzqqbyksg",
                        "djxdgbpuyerxgrmx",
                        "flpmjcetsinyjimc",
                        "fvflhdedljqrcqle",
                        "kwxjejihbgmtnagf",
                        "rzjssfxzzoddvgdc",
                        "squxtuwvjnzbhzsc",
                        "swacqepcxnosmcll",
                        "udlfdefgndowttah",
                        "vjarzxzevsdnftcl",
                        "wrvfnbtdqgpsnzic",
                        "zqswdfwtkyehitft",
                        "degvuccboupdnasm",
                        "tafluhgrtixdlhpv",
                        "tmikjfqekaorgssv",
                        "ufqiiuoxasjwxqbf",
                        "cedgzkylsgxnlcjg",
                        "exutskjkecvotaxd",
                        "mjwrreshlbmzkwmc",
                        "xiesnbkcyzrpzlyq",
                        "zwutaiivgrxnrwat",
                        "bfmdeosllvjkezwq",
                        "hvjwbevmcmjpnknw",
                        "optzzqvbwwriedfo",
                        "pfwcfdvpkuyucnkn",
                        "aaykjdjgdzrrdvxz",
                        "blmjcblhzfqwhgew",
                        "qjkwsppqbsgsvjwa",
                        "qxksnnsrnebfkwqs",
                        "skwelgffvlzgmbro",
                        "tdldeeccsirqwpcj",
                        "enmicxqiumbpozpk",
                        "gtbzqhsuzzdfhzfv",
                        "gxgjyxrnnugizdvf",
                        "wycspyzpbmhbnmda",
                        "cfuyjykoohewxzeg",
                        "kilbdkfbpczjrqek",
                        "vnefzhazthgsjuax",
                        "wkzwidzltxinpgen",
                        "dmqhptvycdmkaxbw",
                        "gdzfmtghobzpihgc",
                        "vomuvsgbhqzjwhgb",
                        "vxqfscklywhurrjp",
                        "yxovdmyzjzoutcek",
                        "abipwhwqnzenjxfn",
                        "acvypvzmenxkevbm",
                        "admgymnmeilfhmji",
                        "aewtczgiyochvagl",
                        "afmufyguudlwbcix",
                        "aggyqhwjksgqtxdd",
                        "ajdmkzcduerbdsww",
                        "ajvhjkzguyeszaqp",
                        "akqknybjyxwbdpot",
                        "aloltvlyufzyxfvg",
                        "anwpfxivfvhnobvz",
                        "aoytjdcfreqvurza",
                        "arfkjhowhuqewzvc",
                        "asbtrxjnhqdpazot",
                        "atsglyxkfbaztzlj",
                        "avrwlknteymnpjpk",
                        "baqjsealekltnrgg",
                        "bawsoqdugnynetyj",
                        "bdmklueoovgkajff",
                        "bfmvfelwblrzqfyr",
                        "bgbhznmwwidntzab",
                        "bgqrpfiflzijywyu",
                        "bikffjqejohkyhat",
                        "bkwszkqrqybfgpyn",
                        "bowuhkfextvyabch",
                        "bpqxbrvavqshzebb",
                        "bpuzzsqfyvebjzjg",
                        "bqcnaxkvbmfieysy",
                        "brjgjnnpueqkyaxo",
                        "buuihjqtdgilqzjc",
                        "bvfbihgnteuiuaov",
                        "bvkytcvosbaunupg",
                        "bwjkokfezucsuigb",
                        "bxksiwcqwmxjcbci",
                        "bxrkvmsmoqvefhra",
                        "bxzfdlphpiwyjeys",
                        "bzsxlzwfqbnmljsm",
                        "caovvakxarqpgymh",
                        "ccxwaznvwtdltwlt",
                        "cedczcxvthqqkwvn",
                        "ceswaufhjtmqcndn",
                        "ckxqqcnqrqxijmmf",
                        "clcqzivttlcdfpnv",
                        "cllupxtcyclounsg",
                        "cnvpgiyrcrbsvtxo",
                        "cpixpqtyjwdgmldj",
                        "cwrigmmyfzesuezf",
                        "cwshqcgmaazzefkx",
                        "cwxtybsrimchiwdv",
                        "cxvltpchlhlatjkb",
                        "cyftaexytlgvmcbd",
                        "degdppvcniqrzruc",
                        "disoykeofihapsal",
                        "dkgrgmlhhtnvzmps",
                        "dlemjwpmokwptnai",
                        "dluodrxtjdtvbxug",
                        "dlwcludeemsmffyb",
                        "dqmsefrpxrwielmk",
                        "dqxenajfgcimjgnw",
                        "dvmnbbkcvcgwnaen",
                        "dxoirhatawazqmey",
                        "dychjlsxfaurgode",
                        "dzjyqrdmawtdcqbx",
                        "efhjvgwyjfjqsdna",
                        "efyukbppkfgttvvw",
                        "egsfpimnisvvfkne",
                        "ekwqttgkaobektch",
                        "ekztjicqomhuclqr",
                        "elvboiqxkxwhtgzg",
                        "eootycnsxmeekotz",
                        "erouvyhobhzcycuk",
                        "esiuazjovwvdlgjy",
                        "esvszhlxzbxeecme",
                        "etegzqakpcvyhkaj",
                        "evuqnfndofizyoqn",
                        "exfftzvkfnajarkm",
                        "eyrukxfjgrcdrqeo",
                        "ezjnsjxvhnocwwix",
                        "fbvdqkwltwgykywc",
                        "feioipyfbkxhcsyq",
                        "fhliexbdvrlrpjvx",
                        "fijtohsiakkeuuct",
                        "fjzqkqcjerkjykkk",
                        "fliymzbupomtmyry",
                        "flnipmkwonjnaqsp",
                        "fnfpmchfyyqmdtfm",
                        "fouvkndsdstwjqpj",
                        "fozvmjndontqoxpg",
                        "frdityocokfyohoa",
                        "frvooqzltrzlbhxb",
                        "fszxbpjtsihsmnqv",
                        "fupocenmkiiluzpe",
                        "fvrvkxucfyuyfpbk",
                        "fvsyahnxhitfllgt",
                        "fwqrokhhbukfpssj",
                        "fydlanmzkobgcfsj",
                        "fzcjreusldmxavjy",
                        "fzgvfpmdmggikezp",
                        "gforiqpfasfwlkfl",
                        "ggqsqgrasnpkxano",
                        "gicokqmbjnafngon",
                        "gkniccewzkphqzrp",
                        "godkpvbnbdeseoct",
                        "goropquvqaoaajrk",
                        "grpzbvvgujnswyyg",
                        "gspurupoewenqznk",
                        "gvaasolsbmnbjhah",
                        "gvordmjbkxszftsl",
                        "gvsbsfrfcvftmytm",
                        "gvxirlwrjrrnoadg",
                        "gykwyopsdhbsalvd",
                        "gzpmemdiurffxomf",
                        "haowzcsrftoqsrvi",
                        "hgyoclvrybybkocm",
                        "hhidavhckwcwznhf",
                        "hikofhdgvhuwkixj",
                        "hixbnwflcimyepla",
                        "hjejiuqyfrvtxagi",
                        "hjhlpxkdgqzdlnkc",
                        "hjhvhzfpslejsnej",
                        "hotinomqpajebeov",
                        "hqixaqcgdcbagrmw",
                        "hrlyreijarvikmlk",
                        "hruelqcyvmwzsqkp",
                        "hungxfwbkelospfy",
                        "hwsgwbkydspkbben",
                        "iadmwbxpppukpjyh",
                        "iadwyxxyvkcpyeus",
                        "ibjlpnapcnsmgugu",
                        "iexbeucevqnjjbcz",
                        "ijxmcnthqquddvhc",
                        "infvsqmvfzjpyfae",
                        "ipyrvtdugjovdwzv",
                        "itmcxdqtvddvmanj",
                        "itvlnddnkkmyemme",
                        "ixyvsrnksxeiqbve",
                        "iydbustazndekvfq",
                        "jancrvhjhcbxreda",
                        "jbvhqxmbarxynmfk",
                        "jcpjlgfslytgmbjq",
                        "jcxkvyjnzflnlzvh",
                        "jdsmqjpfexexznya",
                        "jghfkxkawqeujuhj",
                        "jhafhnhmasllifix",
                        "jiyhnfvmyyrpnzyx",
                        "jlhzkuikphkxcigk",
                        "jmlbcbnedxdoagqm",
                        "jskghzhjrpywrbfn",
                        "jtedxzwqoodxzcaq",
                        "jxlbmlxexeucwbue",
                        "jynbrbzntxrssxzh",
                        "kbixxyjwgxmbhcsa",
                        "kbnrpawcssaxrpmb",
                        "kbqauyzezmwspqvv",
                        "kfvusykzaeetiqtt",
                        "kglgveumqmtwrqsf",
                        "khwbllfppvhgkgzc",
                        "khzmqnkqbaqvnakh",
                        "kjogjnoblzpoxgyr",
                        "knrylcwjpefiqlma",
                        "kpnwdujiylvsiuhp",
                        "ktpoqrjuewxmkjqr",
                        "ktytfazsvecrjvzl",
                        "kumhekfclnypkavw",
                        "kxmtwjjyzuqqgmjw",
                        "kzhhwebpekxgvfsl",
                        "kzzakxocsxhkvslf",
                        "lctcvcvytpesgryp",
                        "ldnocwfyeejbmmcy",
                        "lhamctzhosdtmdix",
                        "lhgeydlzsntbaqzj",
                        "ljwfegchielwaghb",
                        "loomciwexxewgiut",
                        "lqohoawdpvdisdiw",
                        "lqsgdewyevczcvwf",
                        "luxhsezouvtbkbpn",
                        "lvpcmycoagwxqpag",
                        "lwclhevnunilhrmm",
                        "lwhjrctubjkbhzmu",
                        "lwtlsafdbhymtibi",
                        "lwwzmxipnntydwir",
                        "lxchmlyoaiocynox",
                        "lxhecyqzfsucxgqm",
                        "lybpmhaivmaqtmsq",
                        "lzsfpyidvnkaxnvs",
                        "mbnozlcufjgvpcdb",
                        "mdiqmxwkzvnpeaop",
                        "mdqyvrtwekmeflye",
                        "mmfquhvxcmjcvmhz",
                        "moayoogjmiizcbez",
                        "mpnamiwsqkvamhfa",
                        "mqsiquclpholncqd",
                        "mqzhmlqqmpafpbqw",
                        "mtcsefxrgtfdqous",
                        "mtubnuteguketfck",
                        "muixzziwtwouzapq",
                        "mxmhlvlmychxzork",
                        "mxytuavlfghapjvu",
                        "mzlcdmigakbbuzli",
                        "ndepxuvlaiqzdnan",
                        "nfmbusxwwqhsaquy",
                        "nfrqxttuhpuqvwti",
                        "nfxbfvlwvmxfproe",
                        "nggwrmvazdxdjyfh",
                        "ngpgrthcqiirdsux",
                        "nhembilpmgrfjifn",
                        "nhmkqmpmstaunzqh",
                        "njcwousmigzpursi",
                        "nkktflvfoasvkvht",
                        "nrwphouoeazzmbmx",
                        "nsgbpbjvswwlhvmm",
                        "ntjpzidotcatossl",
                        "nwfvqtdnlrvhdbuc",
                        "nzanewsbtbnpgrom",
                        "nzgfjmknhxdezggp",
                        "obtymepcippfwigb",
                        "obzgnvzzatnjoryi",
                        "odjkyxbmtxqhkflm",
                        "oeexhaebfkkjfpff",
                        "ogdxwqtrpclsxeyw",
                        "ogyvyvhcaefqrlgk",
                        "oijipbtrzkghftpt",
                        "okzpgwvslpvgceva",
                        "opojibguvnupidif",
                        "oqbjvmfvjonftdxi",
                        "ospbwzzmmxeovscc",
                        "ouhkmefnnchsggpl",
                        "ovhdtvldyrrurawo",
                        "owkgoejsxqlzahbz",
                        "owrozlxfshxrcgvh",
                        "payritakwxpyzwqq",
                        "pbwbzedhenqmpfqt",
                        "pebdztssohmloufw",
                        "pfkfojczxwevqesz",
                        "pfvqxmrnkptcrhet",
                        "pgkgdfabkhkbviht",
                        "pijaubxodtxcsqjp",
                        "pticuqiimwdrkpdy",
                        "pvyfdiggxtjoyhqf",
                        "pyhcuhumhsoodqwl",
                        "qahlidfcpdaofkwm",
                        "qbohnomeacnwdafj",
                        "qdhfzxrzisivuhbx",
                        "qdmbicmyqrqalixj",
                        "qghhvatpvekejzpf",
                        "qmahqrjhkxvkwboe",
                        "qoflnrycwjlbfmow",
                        "qozlaoxmwusgalpz",
                        "qpjdblaqrqyuoaqk",
                        "qppmxxfbqiiallmp",
                        "qukbrubjquwstnyf",
                        "qwedbcvlquqfoycc",
                        "qwqwzvbefvgugtzi",
                        "qwshkzmlvlerxsov",
                        "qxnyigoiwisibpko",
                        "qyqvfzuwfpyztbla",
                        "qzkbvcycbyxrgbqk",
                        "qzrkqxhgbqfyswsj",
                        "rclsneerlfasdcpi",
                        "rgfytoxurocumuxu",
                        "rsphcdnwdddxhdvb",
                        "rtqyfobkpliuutfx",
                        "ruyuflpnypnsgkbq",
                        "ryjiidsxttvdcpwu",
                        "rytmtyltypttvqjs",
                        "sboaeuuuhpsjujpz",
                        "sbrarddcurfhmmqk",
                        "selnccftdsqbiurb",
                        "sghyfposeljrkedw",
                        "sgknghheolfpzuid",
                        "skgvahbwdkddoxha",
                        "smcawzwicovvejgm",
                        "sncpkctrqcditirm",
                        "spqqpwucqcaspwkb",
                        "sqqvhmadjqegpsps",
                        "suycgjdrxxvxgmha",
                        "sxmsrnbwrnvfjcvp",
                        "szluwlsqbkcnchxg",
                        "tbpblaaxsajjlyok",
                        "tcyceqtrfusfmkpy",
                        "tcyogsbbufjzekla",
                        "tdvzvrkldmrkqeth",
                        "tegzsblugaczvdmy",
                        "tgfhgapnsxiewemd",
                        "timtcrwibllgvgxy",
                        "tjdlkefrbysjheap",
                        "tjvewbsfsiqtqttp",
                        "tkomxtfmozdiflzf",
                        "tlrnhgwgduswslyd",
                        "tlspgqlrhuzholye",
                        "tneakanblaxyevhf",
                        "tpzzxliudfwqpopv",
                        "trcsvrxdekscyvyq",
                        "trwedbipujnvnhpr",
                        "tsfyxgkwdidzgzpg",
                        "tsufyplsxyqgndsw",
                        "ttcbkakfxnfsllyq",
                        "ttwsxzxrhwuzystf",
                        "tuudpbartgtwkoms",
                        "txcfordzmkkiicwu",
                        "tzsnmmekuhggblhv",
                        "tzyaldqhudfiajin",
                        "uaweumzfugnsyqmx",
                        "ubixvbvqnnjdksca",
                        "ubmafwgdsmbkfmwe",
                        "uczhoudymbvnhter",
                        "uevoqlbbbmmhkdbi",
                        "ufxhmwifrakfhfmb",
                        "ughzfvxgeziewvdi",
                        "uhhrakuolcaxvsbh",
                        "uhnrmazhzouxvsqd",
                        "ujfqhxynqnqeldes",
                        "ukztcjpqrpetqrnx",
                        "uouhydwldpcdzuoj",
                        "updzeguaxbccwpoe",
                        "urbnjdherequimyo",
                        "ureljkoqqvqbpdvx",
                        "uskrsfpueljrtxkg",
                        "utvgeykupnwzepks",
                        "uudtrowerqhfztjo",
                        "uudulyvocjutaxtj",
                        "uwqsodousnydlsud",
                        "uxpkyjrybttfrluy",
                        "uysllzwmzcsweunu",
                        "vafxeruawxlvrttn",
                        "vbjxuynqachujrmt",
                        "vfprhybczhnkefdf",
                        "vnsxpowqjjomnmac",
                        "vrubhbzjguaxfmlc",
                        "vtgypuzvlawmkolb",
                        "vtmylwmvnssatjlh",
                        "vuqhkgrnmheydqku",
                        "vuvbtxegkdifkviv",
                        "vwerelrvyumnkbwk",
                        "vwjjtxhpebfhzzck",
                        "vzinjyeuiebqjmep",
                        "wffzvseexopfwwjy",
                        "whfzavgfbojmgezm",
                        "wlttiymytfacsrli",
                        "wmmjbitjeevklkzj",
                        "wniogbpezwqrinyt",
                        "wnrwtzbbxbvnmbqd",
                        "wofhkqytrnqvbije",
                        "wotupxkyxwcienzd",
                        "wqqxbrsnrtnuxjjl",
                        "wqsapwecaqwzorqn",
                        "wrpeamdcqpawnqag",
                        "wvnsmznngunxhcsb",
                        "wwmsynqlijbriqxy",
                        "wxcohilpavlwlnze",
                        "wykcypxicfqltavz",
                        "wylaluxuyuqkytus",
                        "xaaujpnniyiuhfql",
                        "xbdjgmfolqdfvftr",
                        "xbhnetrbyfixuzmj",
                        "xcpcxksgiefkqznu",
                        "xcvdausiwfrjukgn",
                        "xieosvyuphbcyzul",
                        "xjsnlswprucbsehn",
                        "xndokrsndaodfknp",
                        "xnwekpoxnvwckfcp",
                        "xokomvoaaiyuedhu",
                        "xqascjfdlnlxubce",
                        "xrmvlihddtxlbzvw",
                        "xrxxucncrqtcgixl",
                        "xsfzlqjhqbcpswcz",
                        "xuxdvcabcanlgmst",
                        "xzdakcdqrnwhtpdb",
                        "ygctwjsvugjhuylz",
                        "yhotwssicqxqetep",
                        "yhzuhoixugafkonm",
                        "ykzumnthkadrzjdb",
                        "yljpdxzmshdpmyhl",
                        "yobesdjweimafxnq",
                        "yohngmkmrueenrvs",
                        "yqztzexmqeyeirmv",
                        "yrfmcopbrlmfinuq",
                        "yrnbnvvghdwvpayv",
                        "ysmwdnymkzsgskpv",
                        "ytckjafhdfppmrhv",
                        "ytornvfpgbsoizqr",
                        "yvuoearellwavkzs",
                        "yweystgylcxxranw",
                        "ywnimisaozuyjomi",
                        "zaiwfnldcznzgrfe",
                        "zayyxxzvekzuooyq",
                        "zbqqfjqpoluazvlo",
                        "zczpxdcxdciitjcu",
                        "zeqmrfiqiusygxdd",
                        "zghfsejpgrfrqfdp",
                        "zhzmkwjsrgvudmmw",
                        "znppxsjvuytcambw",
                        "zqjvnptshpgofkqc",
                        "zrclcvscjwdbabii",
                        "zsuewlbquazyrgvl",
                        "zukemjnabmcizdzn",
                        "zuqrdemwihnexkpw",
                        "zzjxvhegwmgqodzk",
                        "saempmkfulqhwfqk",
                        "uamadghmregezetz",
                        "lxvjgyjdszxtcryf",
                        "ngombkqqomblyxwv",
                        "pyholyswkkqjmxlj",
                        "jcefoutonncubdss",
                        "pvrjjyumueakzstw",
                        "qqmkwgdqaimwcbxo",
                        "rwtwnvhjqabvovnz",
                        "dtpbahjtnmyuxqno",
                        "gpclrtlzecazeeev",
                        "rqaprgqcktgrlxnv",
                        "dnvcqpxxzahdhbvy",
                        "tupmlwnkgjcgcmuv",
                        "bvuzvpriwqlnbjxt",
                        "iemmvtjtejhlteqa",
                        "dhjmmmtnpcnalzna",
                        "xjaddkudsebowzen",
                        "xqydufxhniyjunrl",
                        "yzdbjmwwtofxmpaz",
                        "nxwedpnhirijkodc",
                        "nhoebceeiacnmvym",
                        "gmzbnaysqjpkzqbt",
                        "kmlnlefquqpparsa",
                        "ptbudvgjgycmmsdq",
                        "ohxrgpugowiyinhv",
                        "xpxsjmglcvcsxwdy",
                        "ogrmvnhwyeydwcxi",
                        "rulqevsymrlwrsrz",
                        "gkxcvooedomgcagl",
                        "phprbhssfhrtbeue",
                        "xoetbcowyxwukawr",
                        "ejeggxbwhufjtjhd",
                        "lfzbrhthlxhnmhva",
                        "adzzjitkyqlberpu",
                        "yxzgvihpyqafgdmy",
                        "qcykqtxlqnbcqfct",
                        "zedpyrfkmhzqxmaz",
                        "htedybhazfjiueyj",
                        "gsooyxmnwsucrksh",
                        "grnnfnsjjydskrht",
                        "ldkzuxzespcgajev",
                        "nmkzmncfytfwyfvt",
                        "szduoosmrfqduakm",
                        "vokpwtikxckeemdi",
                        "xkqlmdeookbxzzhv",
                        "pustczakchcimwuy",
                        "vbnagxwgmwirhnjt",
                        "demgvtbzilochupd",
                        "whcqrtwarljaqocm",
                        "rjhfsrwtoqfqvuqu",
                        "eslneidrjqwzpqhd",
                        "gcmwblighdilwauf",
                        "hvziklxqbjbvncjy",
                        "lhyhsxrxdftbsavk",
                        "mpwepwxyokmciojj",
                        "njujuhbmnqusynwf",
                        "oqkxqgmcsytmcsjz",
                        "uzfgpmnazksmudrw",
                        "kcjttmlajpvbntkn",
                        "lwrjcljtxkokvnes",
                        "qbztetcodwhfmoyg",
                        "ohgtowaarzphsifb",
                        "lqalilfrsznnxarm",
                        "vxumxjoeywcphfoo",
                        "cbmmnlpqoyyursux",
                        "exkqtrkthhgvjqdl",
                        "ejbxcyhffvcouoxd",
                        "kkxluqnhrmwkfqnh",
                        "ltdxvujhaocpnmzf",
                        "mymdahqxtsywqpdn",
                        "wrzuftzqwoiwsmfc",
                        "gctieesvmkeoozqx",
                        "mjpgppxzelxrbcnt",
                        "wiyyghhcxezudyxg",
                        "xzwnliotgalpusga",
                        "dohofttmidfqjozb",
                        "uzrhbfduaqijosql",
                        "xebooruxiuwbpzdc",
                        "cpruzckbhhcyorgf",
                        "iwxvflrheripbuvw",
                        "mcuawemlwwgaiesn",
                        "vqdlslwzvwucentl",
                        "oihtzffwsrwsjnfu",
                        "ypgcfauffeqpeerz",
                        "dhxftxnxtxlgqcqb",
                        "bwpieeluivljdtai",
                        "gyhebbdhtmqwwxnp",
                        "zfylmujpvzgqqfxo",
                        "tksyxmdgogmokuxv",
                        "yglorajvvrsviget",
                        "cgighhnwnkxluccz",
                        "vudjowytbogxkrcy",
                        "xabkvgvnbqzrmnyc",
                        "zrfduayrhhofpqtt",
                        "clwswcgzlaojjddv",
                        "fjimpbebyszdttpl",
                        "twhtulgwsricneea",
                        "ixfiagqhmszowdmf",
                        "jkhjcfudwqurdoex",
                        "zvdpnzgvkjkoophv",
                        "uwwrbkmjbjyxutfq",
                        "zzlzzujtugbfpsvv",
                        "ssnqyyteovyaxylf",
                        "nofmcfnaiuzlqgrk",
                        "qrgsdbjbjwwgirvo",
                        "ybbdbpunwekygnto",
                        "gujwvdfcmmqcwxfi",
                        "cjcthmigqkejxuzi",
                        "tsvmlrkmftqbjvub",
                        "kzqcxkrdytalrphb",
                        "fnqgfjfkzhfbiicl",
                        "beagnicqcxahqkeq",
                        "mcloznejvtelpcan",
                        "qachmbxcslsazphb",
                        "tcnfpudadgannoey",
                        "ifrzhyqsimoeljaa",
                        "uzqtsirvtxcfqnbp",
                        "ylboorftnzombypn",
                        "ucggoqoneixjlxxy",
                        "owxrlgxbigikfgtm",
                        "qwcrrrebwyeauczj",
                        "uclvvrkbezlvaulu",
                        "adgfkcvmsaxxghoc",
                        "pqpqthiapbycbhor",
                        "sfcciovhmwqehacv",
                        "jkwlqsmedtplrvtj",
                        "kvcddisqpkysmvvo",
                        "iklmkdrwatltidff",
                        "esilvarzflhfmjhh",
                        "gxpuiivthwcmpcmc",
                        "hjyumbyuzbeubtbb",
                        "jkarjtlhihuxqzfm",
                        "anrwlguztftzfdng",
                        "drpwkafcvcypyrmw",
                        "fklewvbxuecmupxn",
                        "zagpnfpbwgeyeufr",
                        "aocfhyagfzdywcih",
                        "sdvssyrvwfwmdccl",
                        "wcyahxrmwqvhmadq",
                        "gzebcnjcmqioqcjb",
                        "ggadbhlnfgoflkaf",
                        "usjukvawgoqplrph",
                        "kwuuuvwdrjkyqyfv",
                        "cmjjolnwfprpzntz",
                        "clmcokjtplrbzvuh",
                        "vpvsuuudxglarezp",
                        "xoueplwxwxrzasti",
                        "jakvzvdollijyhwm",
                        "dbtkrhmbfxpkqbau",
                        "fadjogsnmecatcfb",
                        "ivhhwynrahlruefk",
                        "ysonsqntnqnqagnn",
                        "qdsjznqzjxlekjtp",
                        "ujwlwswdwvbpacnf",
                        "ufpwnqycocwwbgqi",
                        "uzmldekmvczimsrj",
                        "tgvulwtrjyegawlr",
                        "ehtbxdjhvcwdapsg",
                        "swxgkelaxkoffszz",
                        "ambmbeydwsdljdcc",
                        "pyykjiriqrhjduly",
                        "jepialiqqsttgcid",
                        "fpfzaadmykntrupr",
                        "gjxmrfgnorpfspbb",
                        "iqepotyqjqeebzix",
                        "lmqoiaqyftqublmk",
                        "ipauahivutejsrev",
                        "hywzsmogbhnfcaxk",
                        "yybjogamsfqljfpu",
                        "sioekxjbocpzrjzi",
                        "taovawittfogygzi",
                        "vpzzxdehhwlzsgrp",
                        "isyektlfmcpmotpl",
                        "gapclpflkdsbeorm",
                        "ggaokfjtqxyctvok",
                        "svcvmlpsqtzbrmnz",
                        "szvfwsizhxrbklhz",
                        "stgeqvsewqntykyo",
                        "rlkrrmxxdgaxangi",
                        "ggidexivtrafqwem",
                        "johsjccpkithubii",
                        "kcfhiwouwwfjqtta",
                        "shemwbbeliuvnvvm",
                        "zxvcbwcwoqnkxxbs",
                        "nyrtstlobluggnkw",
                        "rwrevaiebpmviwqz",
                        "coufviypetbrtevy",
                        "cgkclpnidlmetsrb",
                        "ldxjynecsqlswvbq",
                        "ixwsqebjjdlxcqsq",
                        "hwldevoubgzgbhgs",
                        "iknapxqudqotqiig",
                        "dpcnodgqfivkhxvn",
                        "xyuwuxlpirkzkqdb",
                        "dwsasdexwmpsmowl",
                        "kzwthrslljkmbqur",
                        "epbwnmcyogpybxlm",
                        "rthsjeyjgdlmkygk",
                        "yzxgnwgpnrdprtbh",
                        "jhdjdpthkztnjvmb",
                        "gjbalugsikirqoam",
                        "wpsyqubfrhdspxkx",
                        "wyqgeeclrqbihfpk",
                        "hpohizpkyzvwunni",
                        "iulvirmzdntweaee",
                        "nhwgapjtnadqqaul",
                        "aparvvfowrjncdhp",
                        "zoypfizhpbtpjwpv",
                        "dllcylnkzeegtsgr",
                        "rrqbtdjvuwwxtusj",
                        "jgkpiuuctpywtrlh",
                        "pjbnwqhnqczouirt",
                        "biqzvbfzjivqmrro",
                        "zqruwnlzuefcpqjm",
                        "prtnwsypyfnshpqx"),
            vh_mm_ord = c(0.997442455242967,
                    0.994884910485934,
                    0.9923273657289,
                    0.989769820971867,
                    0.987212276214834,
                    0.9846547314578,
                    0.982097186700767,
                    0.979539641943734,
                    0.976982097186701,
                    0.974424552429668,
                    0.971867007672634,
                    0.969309462915601,0.966751918158568,
                    0.964194373401535,
                    0.961636828644501,
                    0.959079283887468,
                    0.956521739130435,
                    0.956521739130435,
                    0.953964194373402,
                    0.951406649616368,
                    0.948849104859335,
                    0.946291560102302,
                    0.943734015345269,
                    0.941176470588235,
                    0.938618925831202,
                    0.936061381074169,
                    0.933503836317136,
                    0.930946291560102,
                    0.928388746803069,
                    0.925831202046036,
                    0.923273657289003,
                    0.920716112531969,0.918158567774936,
                    0.915601023017903,
                    0.91304347826087,
                    0.91304347826087,
                    0.910485933503836,
                    0.907928388746803,
                    0.90537084398977,
                    0.902813299232737,
                    0.900255754475703,
                    0.89769820971867,
                    0.895140664961637,
                    0.892583120204604,
                    0.89002557544757,
                    0.887468030690537,
                    0.884910485933504,0.882352941176471,
                    0.879795396419437,
                    0.877237851662404,
                    0.874680306905371,
                    0.872122762148338,
                    0.869565217391304,
                    0.867007672634271,
                    0.864450127877238,
                    0.861892583120205,
                    0.859335038363171,
                    0.856777493606138,
                    0.854219948849105,
                    0.851662404092072,
                    0.849104859335038,
                    0.846547314578005,
                    0.843989769820972,
                    0.841432225063939,
                    0.838874680306905,
                    0.836317135549872,
                    0.833759590792839,0.831202046035806,
                    0.831202046035806,
                    0.831202046035806,
                    0.828644501278772,
                    0.826086956521739,
                    0.823529411764706,
                    0.820971867007673,
                    0.818414322250639,
                    0.815856777493606,
                    0.813299232736573,
                    0.81074168797954,
                    0.808184143222506,
                    0.805626598465473,
                    0.80306905370844,
                    0.800511508951407,
                    0.797953964194373,
                    0.79539641943734,0.792838874680307,
                    0.790281329923274,
                    0.78772378516624,
                    0.785166240409207,
                    0.785166240409207,
                    0.782608695652174,
                    0.780051150895141,
                    0.777493606138107,
                    0.774936061381074,
                    0.772378516624041,
                    0.769820971867008,
                    0.767263427109974,
                    0.764705882352941,
                    0.762148337595908,
                    0.759590792838875,
                    0.757033248081841,
                    0.754475703324808,
                    0.751918158567775,
                    0.749360613810742,0.746803069053708,
                    0.746803069053708,
                    0.744245524296675,
                    0.741687979539642,
                    0.739130434782609,
                    0.736572890025575,
                    0.736572890025575,
                    0.734015345268542,
                    0.731457800511509,
                    0.728900255754476,
                    0.726342710997442,
                    0.723785166240409,
                    0.721227621483376,
                    0.718670076726343,
                    0.716112531969309,
                    0.713554987212276,
                    0.710997442455243,
                    0.70843989769821,
                    0.705882352941177,0.703324808184143,
                    0.70076726342711,
                    0.698209718670077,
                    0.695652173913043,
                    0.69309462915601,
                    0.690537084398977,
                    0.687979539641944,
                    0.685421994884911,
                    0.682864450127877,
                    0.680306905370844,
                    0.677749360613811,
                    0.675191815856777,
                    0.672634271099744,
                    0.670076726342711,
                    0.667519181585678,
                    0.667519181585678,
                    0.664961636828645,
                    0.662404092071611,0.659846547314578,
                    0.659846547314578,
                    0.657289002557545,
                    0.657289002557545,
                    0.654731457800512,
                    0.654731457800512,
                    0.652173913043478,
                    0.649616368286445,
                    0.647058823529412,
                    0.647058823529412,
                    0.644501278772379,
                    0.641943734015345,
                    0.639386189258312,
                    0.636828644501279,
                    0.634271099744246,
                    0.631713554987212,
                    0.629156010230179,
                    0.629156010230179,
                    0.626598465473146,
                    0.624040920716113,0.621483375959079,
                    0.618925831202046,
                    0.616368286445013,
                    0.61381074168798,
                    0.611253196930946,
                    0.608695652173913,
                    0.60613810741688,
                    0.603580562659846,
                    0.601023017902813,
                    0.59846547314578,
                    0.595907928388747,
                    0.593350383631714,
                    0.59079283887468,
                    0.588235294117647,
                    0.585677749360614,
                    0.583120204603581,0.580562659846547,
                    0.578005115089514,
                    0.575447570332481,
                    0.572890025575448,
                    0.570332480818414,
                    0.567774936061381,
                    0.565217391304348,
                    0.562659846547315,
                    0.560102301790281,
                    0.557544757033248,
                    0.554987212276215,
                    0.552429667519182,
                    0.549872122762148,
                    0.549872122762148,
                    0.549872122762148,
                    0.547314578005115,
                    0.547314578005115,
                    0.544757033248082,
                    0.544757033248082,
                    0.544757033248082,0.542199488491049,
                    0.539641943734015,
                    0.537084398976982,
                    0.534526854219949,
                    0.531969309462916,
                    0.531969309462916,
                    0.529411764705882,
                    0.526854219948849,
                    0.524296675191816,
                    0.524296675191816,
                    0.521739130434783,
                    0.519181585677749,
                    0.516624040920716,
                    0.516624040920716,
                    0.514066496163683,
                    0.51150895140665,
                    0.508951406649616,
                    0.506393861892583,0.50383631713555,
                    0.50383631713555,
                    0.501278772378517,
                    0.498721227621483,
                    0.49616368286445,
                    0.493606138107417,
                    0.491048593350384,
                    0.48849104859335,
                    0.48849104859335,
                    0.485933503836317,
                    0.483375959079284,
                    0.480818414322251,
                    0.480818414322251,
                    0.478260869565217,
                    0.475703324808184,
                    0.473145780051151,0.470588235294118,
                    0.470588235294118,
                    0.470588235294118,
                    0.468030690537084,
                    0.465473145780051,
                    0.462915601023018,
                    0.462915601023018,
                    0.460358056265985,
                    0.457800511508951,
                    0.455242966751918,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.452685421994885,
                    0.450127877237852,
                    0.447570332480818,
                    0.445012787723785,
                    0.442455242966752,
                    0.439897698209719,
                    0.437340153452685,
                    0.434782608695652,
                    0.434782608695652,
                    0.432225063938619,
                    0.429667519181586,
                    0.427109974424552,
                    0.424552429667519,
                    0.421994884910486,
                    0.421994884910486,
                    0.419437340153453,
                    0.416879795396419,0.414322250639386,
                    0.411764705882353,
                    0.40920716112532,
                    0.406649616368286,
                    0.404092071611253,
                    0.40153452685422,
                    0.398976982097187,
                    0.398976982097187,
                    0.398976982097187,
                    0.396419437340153,
                    0.39386189258312,
                    0.391304347826087,
                    0.388746803069054,
                    0.38618925831202,
                    0.38618925831202,0.38618925831202,
                    0.383631713554987,
                    0.381074168797954,
                    0.378516624040921,
                    0.375959079283887,
                    0.373401534526854,
                    0.373401534526854,
                    0.370843989769821,
                    0.368286445012788,
                    0.365728900255754,
                    0.365728900255754,
                    0.363171355498721,
                    0.363171355498721,
                    0.363171355498721,
                    0.363171355498721,
                    0.360613810741688,
                    0.358056265984655,
                    0.355498721227621,
                    0.352941176470588,0.350383631713555,
                    0.347826086956522,
                    0.345268542199488,
                    0.345268542199488,
                    0.345268542199488,
                    0.345268542199488,
                    0.345268542199488,
                    0.345268542199488,
                    0.345268542199488,
                    0.342710997442455,
                    0.340153452685422,
                    0.337595907928389,
                    0.335038363171355,
                    0.332480818414322,
                    0.329923273657289,
                    0.327365728900256,
                    0.327365728900256,
                    0.324808184143222,
                    0.322250639386189,
                    0.322250639386189,0.322250639386189,
                    0.322250639386189,
                    0.319693094629156,
                    0.319693094629156,
                    0.317135549872123,
                    0.317135549872123,
                    0.314578005115089,
                    0.314578005115089,
                    0.312020460358056,
                    0.309462915601023,
                    0.309462915601023,
                    0.309462915601023,
                    0.309462915601023,
                    0.30690537084399,
                    0.30690537084399,
                    0.304347826086957,
                    0.301790281329923,
                    0.301790281329923,0.29923273657289,
                    0.296675191815857,
                    0.296675191815857,
                    0.294117647058824,
                    0.29156010230179,
                    0.289002557544757,
                    0.289002557544757,
                    0.286445012787724,
                    0.286445012787724,
                    0.283887468030691,
                    0.281329923273657,
                    0.281329923273657,
                    0.281329923273657,
                    0.278772378516624,
                    0.276214833759591,
                    0.273657289002558,
                    0.271099744245524,
                    0.268542199488491,0.265984654731458,
                    0.263427109974425,
                    0.260869565217391,
                    0.260869565217391,
                    0.258312020460358,
                    0.255754475703325,
                    0.253196930946292,
                    0.250639386189258,
                    0.248081841432225,
                    0.245524296675192,
                    0.242966751918159,
                    0.242966751918159,
                    0.242966751918159,
                    0.240409207161125,
                    0.237851662404092,
                    0.235294117647059,
                    0.232736572890026,
                    0.230179028132992,
                    0.227621483375959,
                    0.225063938618926,0.222506393861893,
                    0.219948849104859,
                    0.217391304347826,
                    0.214833759590793,
                    0.21227621483376,
                    0.209718670076726,
                    0.207161125319693,
                    0.20460358056266,
                    0.202046035805627,
                    0.199488491048593,
                    0.19693094629156,
                    0.194373401534527,
                    0.191815856777494,
                    0.191815856777494,
                    0.18925831202046,
                    0.186700767263427,0.184143222506394,
                    0.181585677749361,
                    0.179028132992327,
                    0.176470588235294,
                    0.173913043478261,
                    0.171355498721228,
                    0.168797953964194,
                    0.166240409207161,
                    0.163682864450128,
                    0.161125319693095,
                    0.158567774936061,
                    0.156010230179028,
                    0.153452685421995,
                    0.150895140664962,
                    0.148337595907928,
                    0.145780051150895,
                    0.143222506393862,
                    0.140664961636829,
                    0.138107416879795,
                    0.135549872122762,0.132992327365729,
                    0.130434782608696,
                    0.127877237851662,
                    0.127877237851662,
                    0.125319693094629,
                    0.122762148337596,
                    0.120204603580563,
                    0.117647058823529,
                    0.115089514066496,
                    0.112531969309463,
                    0.10997442455243,
                    0.107416879795396,
                    0.104859335038363,
                    0.10230179028133,
                    0.0997442455242967,
                    0.0971867007672634,
                    0.0946291560102302,
                    0.0920716112531969,
                    0.0895140664961637,
                    0.0869565217391304,
                    0.0843989769820972,
                    0.0818414322250639,
                    0.0792838874680307,
                    0.0767263427109974,
                    0.0741687979539642,
                    0.0716112531969309,
                    0.0690537084398977,
                    0.0664961636828645,
                    0.0639386189258312,
                    0.061381074168798,
                    0.0588235294117647,
                    0.0562659846547315,
                    0.0537084398976982,
                    0.051150895140665,
                    0.0485933503836317,
                    0.0460358056265985,
                    0.0434782608695652,
                    0.040920716112532,
                    0.0383631713554987,
                    0.0358056265984655,
                    0.0332480818414322,
                    0.030690537084399,
                    0.0281329923273657,
                    0.0255754475703325,
                    0.0230179028132992,0.020460358056266,
                    0.0179028132992327,
                    0.0153452685421995,
                    0.0127877237851662,
                    0.010230179028133,
                    0.00767263427109974,
                    0.0051150895140665,
                    0.00255754475703325,0)
  
  )
  return(vhmm_ordering_train)
}
vhmm_ordering_train <- vh_order_train()

add_vh_order_train <- function(input){
  
  input <- left_join(input,vhmm_ordering_train, by = "vh_make_model")
  
  # fix for new vh that aren't in the group list
  
  input <- input %>% mutate(vh_mm_ord = case_when(is.na(vh_mm_ord) ~ median(vh_mm_ord, na.rm=TRUE),
                                                  TRUE ~as.numeric(vh_mm_ord)
  )
  )
  
  #input <- input[ , !(names(input) %in% "vh_make_model")]
  
  return(input)
}



vh_popularity_train <- function(){
  
  vhmm_pop_train = data.frame(
    stringsAsFactors = FALSE,
    vh_make_model = c("adhoqfsfdpetomvs","ayeiibefzqqbyksg",
                      "cedczcxvthqqkwvn",
                      "cyftaexytlgvmcbd","djxdgbpuyerxgrmx",
                      "eivjhovgfnfctgjy",
                      "fozvmjndontqoxpg","fupocenmkiiluzpe",
                      "fvflhdedljqrcqle",
                      "hqixaqcgdcbagrmw","kpnwdujiylvsiuhp",
                      "ktpoqrjuewxmkjqr","kumhekfclnypkavw",
                      "luxhsezouvtbkbpn",
                      "lwclhevnunilhrmm","nhmkqmpmstaunzqh",
                      "pvyfdiggxtjoyhqf",
                      "qdhfzxrzisivuhbx","qzrkqxhgbqfyswsj",
                      "ruposftqgswlcyou",
                      "rzjssfxzzoddvgdc","sgknghheolfpzuid",
                      "swacqepcxnosmcll",
                      "ughzfvxgeziewvdi","uudulyvocjutaxtj",
                      "vfprhybczhnkefdf",
                      "wqsapwecaqwzorqn","wrvfnbtdqgpsnzic",
                      "ajktbllxjzfdtwpy","avrwlknteymnpjpk",
                      "bfmvfelwblrzqfyr",
                      "dkgrgmlhhtnvzmps","dohuwjuguzyvqaqg",
                      "eootycnsxmeekotz",
                      "erouvyhobhzcycuk","gbkevbmczkqhkmoc",
                      "ggqsqgrasnpkxano",
                      "ggzcspiycgszcunf","gkniccewzkphqzrp",
                      "hayciibjzwapccnb",
                      "hikofhdgvhuwkixj","hruelqcyvmwzsqkp",
                      "ioqpncqqlflrjzkj","jskghzhjrpywrbfn",
                      "lxchmlyoaiocynox",
                      "lxjkslpwiofoynao","ospbwzzmmxeovscc",
                      "owrozlxfshxrcgvh",
                      "rrsrcesavzhbjqwk","ufxhmwifrakfhfmb",
                      "vxvmxuncsxygbrzd",
                      "ybsenzutfrjternf","zeqmrfiqiusygxdd",
                      "aggyqhwjksgqtxdd",
                      "bgqrpfiflzijywyu","cictcfpmfdmknnye",
                      "cnvpgiyrcrbsvtxo",
                      "cwrigmmyfzesuezf","efyukbppkfgttvvw",
                      "eyrukxfjgrcdrqeo","gvsbsfrfcvftmytm",
                      "heicadwqfavetjwx",
                      "htppstzpipwjtuia","hungxfwbkelospfy",
                      "infvsqmvfzjpyfae",
                      "itmcxdqtvddvmanj","jtedxzwqoodxzcaq",
                      "lqohoawdpvdisdiw",
                      "lqsgdewyevczcvwf","lwhjrctubjkbhzmu",
                      "lwwzmxipnntydwir",
                      "mdiqmxwkzvnpeaop","mshhupropfijhilz",
                      "obtymepcippfwigb","oijipbtrzkghftpt",
                      "pticuqiimwdrkpdy",
                      "qdmbicmyqrqalixj","qyqvfzuwfpyztbla",
                      "tbpblaaxsajjlyok",
                      "tegzsblugaczvdmy","tsfyxgkwdidzgzpg",
                      "ttwsxzxrhwuzystf",
                      "vwjjtxhpebfhzzck","yrfmcopbrlmfinuq",
                      "zczpxdcxdciitjcu",
                      "zzjxvhegwmgqodzk","admgymnmeilfhmji",
                      "aewtczgiyochvagl",
                      "anwpfxivfvhnobvz","buuihjqtdgilqzjc",
                      "bvfbihgnteuiuaov","ckxqqcnqrqxijmmf",
                      "cllupxtcyclounsg",
                      "dlemjwpmokwptnai","dtdrfrtruyhvbztx",
                      "efhjvgwyjfjqsdna",
                      "ezaffjpqpacrufvd","gdtzpvajphaxanpi",
                      "gtvhxebtkefavzhg",
                      "hgyoclvrybybkocm","hixbnwflcimyepla",
                      "iditakunbaxfjcmc",
                      "ismjlsoibleinjdp","iydbustazndekvfq",
                      "jancrvhjhcbxreda","jghfkxkawqeujuhj",
                      "jixkbeuswaznqplh",
                      "jlibzlturkpyjavf","jxlbmlxexeucwbue",
                      "kdsbtuikoaulynsu",
                      "kgezpfvpmpmdicts","knrylcwjpefiqlma",
                      "lhgeydlzsntbaqzj",
                      "lpwtmtiwkgbwhufg","muixzziwtwouzapq",
                      "mxytuavlfghapjvu",
                      "njcwousmigzpursi","nkktflvfoasvkvht",
                      "odjkyxbmtxqhkflm",
                      "ogdxwqtrpclsxeyw","ogyvyvhcaefqrlgk",
                      "ouhkmefnnchsggpl","ovhdtvldyrrurawo",
                      "pbwbzedhenqmpfqt",
                      "pijaubxodtxcsqjp","qbohnomeacnwdafj",
                      "rhxboadaoyvvgflk",
                      "sxmsrnbwrnvfjcvp","tcyogsbbufjzekla",
                      "tdvzvrkldmrkqeth",
                      "tkomxtfmozdiflzf","tlrnhgwgduswslyd",
                      "tneakanblaxyevhf",
                      "tuudpbartgtwkoms","ubmafwgdsmbkfmwe",
                      "ureljkoqqvqbpdvx","uudtrowerqhfztjo",
                      "vnsxpowqjjomnmac",
                      "vylopbnfewdzeury","wcjrtzzkemciejsz",
                      "whfzavgfbojmgezm",
                      "wmmjbitjeevklkzj","wrpeamdcqpawnqag",
                      "wwmsynqlijbriqxy",
                      "wxcohilpavlwlnze","wylaluxuyuqkytus",
                      "xcpcxksgiefkqznu",
                      "xsfzlqjhqbcpswcz","ytornvfpgbsoizqr",
                      "zaiwfnldcznzgrfe",
                      "znppxsjvuytcambw","zrclcvscjwdbabii",
                      "zuqrdemwihnexkpw","ablxjgbyowxrfxed",
                      "akqknybjyxwbdpot",
                      "aoytjdcfreqvurza","asbtrxjnhqdpazot",
                      "bawsoqdugnynetyj",
                      "bqcnaxkvbmfieysy","brjgjnnpueqkyaxo",
                      "btjxvrgfduskmpts",
                      "bxrkvmsmoqvefhra","bxzfdlphpiwyjeys",
                      "caovvakxarqpgymh",
                      "ccxwaznvwtdltwlt","cxvltpchlhlatjkb",
                      "degdppvcniqrzruc","dlwcludeemsmffyb",
                      "dvshwarqhxfcgwfd",
                      "dxoirhatawazqmey","dxpafctvukcmaqao",
                      "esiuazjovwvdlgjy",
                      "esvszhlxzbxeecme","exfftzvkfnajarkm",
                      "eyaqhofitsegmcwi",
                      "fbvdqkwltwgykywc","fhliexbdvrlrpjvx",
                      "fkltkgzmjnzqzlqv",
                      "fouvkndsdstwjqpj","frdityocokfyohoa",
                      "frvooqzltrzlbhxb",
                      "fzcjreusldmxavjy","fzgvfpmdmggikezp",
                      "gdultxlilvdnuwso","gvaasolsbmnbjhah",
                      "gykwyopsdhbsalvd",
                      "gzpmemdiurffxomf","haowzcsrftoqsrvi",
                      "hjejiuqyfrvtxagi",
                      "hjhvhzfpslejsnej","hrlyreijarvikmlk",
                      "hwsgwbkydspkbben",
                      "iadmwbxpppukpjyh","iadwyxxyvkcpyeus",
                      "ibjlpnapcnsmgugu",
                      "ifalilovsdszxmjm","ixyvsrnksxeiqbve",
                      "jynbrbzntxrssxzh","kfvusykzaeetiqtt",
                      "kglgveumqmtwrqsf",
                      "khwbllfppvhgkgzc","ktrfapbareyzyyyq",
                      "ktytfazsvecrjvzl",
                      "ljwfegchielwaghb","luwiodhzrjjobjlw",
                      "mkbpzddzmalsleud",
                      "mtcsefxrgtfdqous","mtubnuteguketfck",
                      "mzlcdmigakbbuzli",
                      "nfrqxttuhpuqvwti","ngpgrthcqiirdsux",
                      "ntjpzidotcatossl",
                      "nzgfjmknhxdezggp","obzgnvzzatnjoryi",
                      "okeuihmplbxhxceo","okzpgwvslpvgceva",
                      "oqbjvmfvjonftdxi",
                      "payritakwxpyzwqq","pebdztssohmloufw",
                      "pgkgdfabkhkbviht",
                      "pmxjblqhvpwflkwt","qidpxyunryowizua",
                      "qppmxxfbqiiallmp",
                      "qwedbcvlquqfoycc","qwshkzmlvlerxsov",
                      "qxnyigoiwisibpko",
                      "qzkbvcycbyxrgbqk","rsphcdnwdddxhdvb",
                      "rtqyfobkpliuutfx","ryjiidsxttvdcpwu",
                      "rytmtyltypttvqjs",
                      "spqqpwucqcaspwkb","tsufyplsxyqgndsw",
                      "txcfordzmkkiicwu",
                      "tzyaldqhudfiajin","uczhoudymbvnhter",
                      "uevoqlbbbmmhkdbi",
                      "uhnrmazhzouxvsqd","ulrbzlswbgzvmpas",
                      "vafxeruawxlvrttn",
                      "vbjxuynqachujrmt","vtgypuzvlawmkolb",
                      "vtmylwmvnssatjlh",
                      "vuqhkgrnmheydqku","vuvbtxegkdifkviv",
                      "wvnsmznngunxhcsb","wykcypxicfqltavz",
                      "xaaujpnniyiuhfql",
                      "xbdjgmfolqdfvftr","xewlloxrajhpbuwy",
                      "xieosvyuphbcyzul",
                      "xrmvlihddtxlbzvw","yhzuhoixugafkonm",
                      "yobesdjweimafxnq",
                      "zghfsejpgrfrqfdp","zhzmkwjsrgvudmmw",
                      "zycfmwxhaaaxdwpb",
                      "acvypvzmenxkevbm","atsglyxkfbaztzlj",
                      "azxtekfvyycfmnpt","bwjkokfezucsuigb",
                      "bxksiwcqwmxjcbci",
                      "clcqzivttlcdfpnv","cpixpqtyjwdgmldj",
                      "dzjyqrdmawtdcqbx",
                      "feioipyfbkxhcsyq","flnipmkwonjnaqsp",
                      "fnfpmchfyyqmdtfm",
                      "gicokqmbjnafngon","hhidavhckwcwznhf",
                      "hjhlpxkdgqzdlnkc",
                      "hotinomqpajebeov","iexbeucevqnjjbcz",
                      "jbvhqxmbarxynmfk",
                      "jdsmqjpfexexznya","jhafhnhmasllifix",
                      "kxmtwjjyzuqqgmjw","lcokgbxbqigkqzcw",
                      "lhamctzhosdtmdix",
                      "lzsfpyidvnkaxnvs","mbjevmuapzxqjnwg",
                      "mdqyvrtwekmeflye",
                      "mmfquhvxcmjcvmhz","mqsiquclpholncqd",
                      "mxmhlvlmychxzork",
                      "ndepxuvlaiqzdnan","nggwrmvazdxdjyfh",
                      "nrwphouoeazzmbmx",
                      "obkqpwjualnnwgrt","pfkfojczxwevqesz",
                      "pyhcuhumhsoodqwl","qozlaoxmwusgalpz",
                      "qukbrubjquwstnyf",
                      "qwqwzvbefvgugtzi","reolzfmikorzxstf",
                      "sncpkctrqcditirm",
                      "trwedbipujnvnhpr","txgvnaysouvjtkrb",
                      "uaweumzfugnsyqmx",
                      "vpvtqlxqaiejzrqo","vwerelrvyumnkbwk",
                      "vzinjyeuiebqjmep",
                      "wniogbpezwqrinyt","wofhkqytrnqvbije",
                      "xnwekpoxnvwckfcp",
                      "xqascjfdlnlxubce","yhotwssicqxqetep",
                      "yohngmkmrueenrvs","yrnbnvvghdwvpayv",
                      "ytckjafhdfppmrhv",
                      "yvuoearellwavkzs","zqjvnptshpgofkqc",
                      "zzubfikjmmfsxhbn",
                      "afmufyguudlwbcix","baqjsealekltnrgg",
                      "bowuhkfextvyabch",
                      "dzbwjjmruyqxyvms","egsfpimnisvvfkne",
                      "ekwqttgkaobektch",
                      "fijtohsiakkeuuct","kzzakxocsxhkvslf",
                      "rgfytoxurocumuxu","sqqvhmadjqegpsps",
                      "tcyceqtrfusfmkpy",
                      "uhhrakuolcaxvsbh","ujfqhxynqnqeldes",
                      "unlqqlfvajjczyks",
                      "wlttiymytfacsrli","wnrwtzbbxbvnmbqd",
                      "yqztzexmqeyeirmv",
                      "zukemjnabmcizdzn","bikffjqejohkyhat",
                      "clwswcgzlaojjddv",
                      "fjimpbebyszdttpl","gmzbnaysqjpkzqbt",
                      "gvxirlwrjrrnoadg",
                      "kguahfjnmerrbtpp","kmlnlefquqpparsa",
                      "lxhecyqzfsucxgqm","ptbudvgjgycmmsdq",
                      "dluodrxtjdtvbxug",
                      "dnvcqpxxzahdhbvy","nbxjozrynlospbso",
                      "pfvqxmrnkptcrhet",
                      "tupmlwnkgjcgcmuv","xabkvgvnbqzrmnyc",
                      "xxppvuhwmnezefxy",
                      "zrfduayrhhofpqtt","bwpieeluivljdtai",
                      "cgrdxjyaxssrszjz",
                      "ezjnsjxvhnocwwix","gyhebbdhtmqwwxnp",
                      "jcpjlgfslytgmbjq","kbqauyzezmwspqvv",
                      "kowgdytyvjhvcmta",
                      "pvrjjyumueakzstw","qqmkwgdqaimwcbxo",
                      "qzgaezfhutbcnkuf",
                      "tgfhgapnsxiewemd","bgbhznmwwidntzab",
                      "cpruzckbhhcyorgf",
                      "innngarflvbnwntw","iwxvflrheripbuvw",
                      "mcuawemlwwgaiesn",
                      "meratbpknllwoefn","opojibguvnupidif",
                      "qbkipjmisqllqwzy",
                      "vqdlslwzvwucentl","yhcelpjnbxpsmoez",
                      "cnicorpxweynumqk","cnlvybtdupkcwczn",
                      "dqgtaigmpivatpeu",
                      "vdetuihriafhetdl","wiyyghhcxezudyxg",
                      "xcvdausiwfrjukgn",
                      "xzwnliotgalpusga","ysmwdnymkzsgskpv",
                      "zojbyremtnxajomo",
                      "cmmuslxsfluvfyof","gsbeyysssgzgkkuo",
                      "jiyhnfvmyyrpnzyx",
                      "kkxluqnhrmwkfqnh","ltdxvujhaocpnmzf",
                      "mpnamiwsqkvamhfa","mymdahqxtsywqpdn",
                      "noxmlxlzirrxdriv",
                      "nwfvqtdnlrvhdbuc","rclsneerlfasdcpi",
                      "selnccftdsqbiurb",
                      "skgvahbwdkddoxha","uhqwwluaswbuqqjc",
                      "vdwfxzfzbybhsmay",
                      "wrzuftzqwoiwsmfc","ciuxczxwhwbxdkdf",
                      "edlxghhjgpmvhabz",
                      "ejlwzigdhipvpndt","gwptulznqgygeegy",
                      "ieqgavmmxulqlvvl",
                      "lqalilfrsznnxarm","qewzxgvvhqhkfcxe",
                      "tpzzxliudfwqpopv","vjxfnjqgvugwjhia",
                      "xuxdvcabcanlgmst",
                      "ywnimisaozuyjomi","bdmklueoovgkajff",
                      "fvsyahnxhitfllgt",
                      "gcmwblighdilwauf","gvordmjbkxszftsl",
                      "hvziklxqbjbvncjy",
                      "jcxkvyjnzflnlzvh","lhyhsxrxdftbsavk",
                      "mpwepwxyokmciojj",
                      "njujuhbmnqusynwf","oqkxqgmcsytmcsjz",
                      "ukztcjpqrpetqrnx","uzfgpmnazksmudrw",
                      "agowcnternxraavr",
                      "celpzeaubkxaxxbx","fgxxyxcbjkodwcln",
                      "grnnfnsjjydskrht",
                      "ldkzuxzespcgajev","qoflnrycwjlbfmow",
                      "zbprczwzmlxgqykc",
                      "abipwhwqnzenjxfn","ajdmkzcduerbdsww",
                      "dcjzdpoxqvgnjpmi",
                      "dmqhptvycdmkaxbw","flpmjcetsinyjimc",
                      "fszxbpjtsihsmnqv",
                      "gdzfmtghobzpihgc","kwxjejihbgmtnagf",
                      "moayoogjmiizcbez","owkgoejsxqlzahbz",
                      "squxtuwvjnzbhzsc",
                      "udlfdefgndowttah","uxpkyjrybttfrluy",
                      "vjarzxzevsdnftcl",
                      "vomuvsgbhqzjwhgb","vxqfscklywhurrjp",
                      "ygctwjsvugjhuylz",
                      "yxovdmyzjzoutcek","zqswdfwtkyehitft",
                      "bzsxlzwfqbnmljsm",
                      "cfuyjykoohewxzeg","fdbwfjqkichwdebq",
                      "goropquvqaoaajrk","ijxmcnthqquddvhc",
                      "jedhlhdmkdprvyex",
                      "kbixxyjwgxmbhcsa","kilbdkfbpczjrqek",
                      "kjogjnoblzpoxgyr",
                      "qpjdblaqrqyuoaqk","rbxibrjokiihgfjb",
                      "sghyfposeljrkedw",
                      "szlkmablxrjoubla","utvgeykupnwzepks",
                      "uwqsodousnydlsud",
                      "vnefzhazthgsjuax","wkzwidzltxinpgen",
                      "bkwszkqrqybfgpyn",
                      "bpuzzsqfyvebjzjg","elvboiqxkxwhtgzg",
                      "enmicxqiumbpozpk","gtbzqhsuzzdfhzfv",
                      "gxgjyxrnnugizdvf",
                      "ldejndeewhhlcvgc","trcsvrxdekscyvyq",
                      "uouhydwldpcdzuoj",
                      "wycspyzpbmhbnmda","xjsnlswprucbsehn",
                      "aaykjdjgdzrrdvxz",
                      "aloltvlyufzyxfvg","blmjcblhzfqwhgew",
                      "cwshqcgmaazzefkx",
                      "fliymzbupomtmyry","khzmqnkqbaqvnakh",
                      "kzhhwebpekxgvfsl","nhqkbmwihkfvhjxx",
                      "qjkwsppqbsgsvjwa",
                      "qxksnnsrnebfkwqs","ruyuflpnypnsgkbq",
                      "skwelgffvlzgmbro",
                      "smcawzwicovvejgm","tdldeeccsirqwpcj",
                      "wffzvseexopfwwjy",
                      "zayyxxzvekzuooyq","bfmdeosllvjkezwq",
                      "ceswaufhjtmqcndn",
                      "dlbnpwopifytzerl","etegzqakpcvyhkaj",
                      "gforiqpfasfwlkfl",
                      "grpzbvvgujnswyyg","hvjwbevmcmjpnknw",
                      "jmlbcbnedxdoagqm","lwtlsafdbhymtibi",
                      "optzzqvbwwriedfo",
                      "pbrroilhklrifbwq","pfwcfdvpkuyucnkn",
                      "qahlidfcpdaofkwm",
                      "qghhvatpvekejzpf","qgnqfinpenszbzig",
                      "suycgjdrxxvxgmha",
                      "updzeguaxbccwpoe","uskrsfpueljrtxkg",
                      "vrubhbzjguaxfmlc",
                      "wotupxkyxwcienzd","xndokrsndaodfknp",
                      "ykzumnthkadrzjdb","yweystgylcxxranw",
                      "bvkytcvosbaunupg",
                      "cedgzkylsgxnlcjg","cwxtybsrimchiwdv",
                      "dvmnbbkcvcgwnaen",
                      "evuqnfndofizyoqn","exutskjkecvotaxd",
                      "fydlanmzkobgcfsj",
                      "itvlnddnkkmyemme","jlhzkuikphkxcigk",
                      "lctcvcvytpesgryp",
                      "ldnocwfyeejbmmcy","lybpmhaivmaqtmsq",
                      "mbnozlcufjgvpcdb",
                      "mjwrreshlbmzkwmc","nhembilpmgrfjifn",
                      "nnzwevftfeodipkn","owkiszjuntmwilff",
                      "ozpyjjijxdpztngv",
                      "qmahqrjhkxvkwboe","qnesuhpxsptzihzg",
                      "szluwlsqbkcnchxg",
                      "xiesnbkcyzrpzlyq","yljpdxzmshdpmyhl",
                      "zwutaiivgrxnrwat",
                      "ajvhjkzguyeszaqp","bpqxbrvavqshzebb",
                      "degvuccboupdnasm",
                      "disoykeofihapsal","dqmsefrpxrwielmk",
                      "dychjlsxfaurgode","fwqrokhhbukfpssj",
                      "ipyrvtdugjovdwzv",
                      "kbnrpawcssaxrpmb","lvpcmycoagwxqpag",
                      "mqzhmlqqmpafpbqw",
                      "nfxbfvlwvmxfproe","tafluhgrtixdlhpv",
                      "tlspgqlrhuzholye",
                      "tmikjfqekaorgssv","ubixvbvqnnjdksca",
                      "ufqiiuoxasjwxqbf",
                      "xokomvoaaiyuedhu","xzdakcdqrnwhtpdb",
                      "zbqqfjqpoluazvlo",
                      "ifrzhyqsimoeljaa","nzanewsbtbnpgrom",
                      "pqpqthiapbycbhor","unmxysjyilftwsvy",
                      "uzqtsirvtxcfqnbp",
                      "ybmkbrazyartpatx","ylboorftnzombypn",
                      "zsuewlbquazyrgvl",
                      "dlrvgwmumnwcjixm","kcjttmlajpvbntkn",
                      "uysllzwmzcsweunu",
                      "fnqgfjfkzhfbiicl","gqfadgvnztixxbmv",
                      "nmhahirmbvqxhxgg",
                      "vbnagxwgmwirhnjt","zakviitdfvxsgkow",
                      "zcqptmhakcmihiry","jrwemlawxsvnwrxv",
                      "myfrksrutuknkcnq",
                      "qcykqtxlqnbcqfct","zedpyrfkmhzqxmaz",
                      "aivacsqryguqpdib",
                      "arfkjhowhuqewzvc","ujgldnoigollndkj",
                      "uotpflqyvprslxjc",
                      "xsonelzsqbpcodxe","joosvbazdbslkqgx",
                      "tcnfpudadgannoey",
                      "twhtulgwsricneea","qachmbxcslsazphb",
                      "sboaeuuuhpsjujpz",
                      "tksyxmdgogmokuxv","ukqbsscpgbfatdhs",
                      "vgaanttvdscmqmjr","yglorajvvrsviget",
                      "blcuqlgntjavsyhs",
                      "jcefoutonncubdss","mcloznejvtelpcan",
                      "oihtzffwsrwsjnfu",
                      "snpaaoiipfuxmvol","ypgcfauffeqpeerz",
                      "beagnicqcxahqkeq",
                      "cufklbvsirnawzmv","dohofttmidfqjozb",
                      "kqxycgbergacgcei",
                      "lxvjgyjdszxtcryf","uzrhbfduaqijosql",
                      "xrxxucncrqtcgixl","xsmqbeukcqahbfgl",
                      "gctieesvmkeoozqx",
                      "loomciwexxewgiut","mjpgppxzelxrbcnt",
                      "rqklbykswxeuovdn",
                      "xgikerzyofvqsmnt","ztwsplndgicacmuu",
                      "aawqanlavsjfqrne",
                      "cjcthmigqkejxuzi","dqxenajfgcimjgnw",
                      "fjzqkqcjerkjykkk",
                      "sbrarddcurfhmmqk","tsvmlrkmftqbjvub",
                      "ttcbkakfxnfsllyq",
                      "ugwrafmvdavbsrzl","vxumxjoeywcphfoo",
                      "ygqjgwzgeierkcpj","drptidaltxzxopwv",
                      "kqubvdyyovhfxtpc",
                      "jkguypwgxebmtnkx","nmkzmncfytfwyfvt",
                      "nolayrxwnjwzgtoo",
                      "qrgsdbjbjwwgirvo","suajnmrxuunoyngf",
                      "szduoosmrfqduakm",
                      "tjvewbsfsiqtqttp","ueujqvpwszzhovbj",
                      "vokpwtikxckeemdi",
                      "xkqlmdeookbxzzhv","ekztjicqomhuclqr",
                      "giyhzprslgbwsaeu","godkpvbnbdeseoct",
                      "huoicgalccftwyvz",
                      "mpmchhrcazhsvjgc","oeexhaebfkkjfpff",
                      "ssnqyyteovyaxylf",
                      "tzsnmmekuhggblhv","yxzgvihpyqafgdmy",
                      "cqewccykrcmvawlo",
                      "dqqtizjjhjmqdqqb","gkxcvooedomgcagl",
                      "hcoxxbfccserxklx",
                      "ixfiagqhmszowdmf","jkhjcfudwqurdoex",
                      "phprbhssfhrtbeue",
                      "timtcrwibllgvgxy","uxfxpraspeoqtmbg",
                      "xoetbcowyxwukawr","zvdpnzgvkjkoophv",
                      "cuxaapvakeemmbaa",
                      "ggadbhlnfgoflkaf","qnixeczkijjyiprb",
                      "wchxrbrhstsmhdsk",
                      "ohxrgpugowiyinhv","dhxftxnxtxlgqcqb",
                      "sdvssyrvwfwmdccl",
                      "wcyahxrmwqvhmadq","aocfhyagfzdywcih",
                      "gpclrtlzecazeeev",
                      "wqqxbrsnrtnuxjjl","gguphuccgeqyojbl",
                      "cbmmnlpqoyyursux","exkqtrkthhgvjqdl",
                      "nzxlhibmhrtafeav",
                      "spbjeokdemicpdey","ucggoqoneixjlxxy",
                      "vpvsuuudxglarezp",
                      "eudwptcohxaazhpt","tkqxtjbbrzagooya",
                      "uamadghmregezetz",
                      "ebdcmhmtqnfkaalo","fvrvkxucfyuyfpbk",
                      "jsudrcgsrfddwixw",
                      "rjhfsrwtoqfqvuqu","xbhnetrbyfixuzmj",
                      "xhczitnzxmxxebeq",
                      "clsrzyechukbaeat","drpwkafcvcypyrmw",
                      "urullqqbaabxllxl","anrwlguztftzfdng",
                      "cqdmtwkacajclcml",
                      "ybbdbpunwekygnto","tyktnjdtbrucursh",
                      "jeckddxjsdolnuhe",
                      "cgighhnwnkxluccz","dhjmmmtnpcnalzna",
                      "gxpuiivthwcmpcmc",
                      "hjyumbyuzbeubtbb","iljhlfeengkciosq",
                      "tduddcyerrjazjsh",
                      "iklmkdrwatltidff","xuioboiuzasnmuva",
                      "adgfkcvmsaxxghoc","jkwlqsmedtplrvtj",
                      "xebooruxiuwbpzdc",
                      "arcsdpohuzvikyaw","nsgbpbjvswwlhvmm",
                      "sfcciovhmwqehacv",
                      "dsqmtbudvjtnnjwq","xjsozzwcppavldee",
                      "xoueplwxwxrzasti",
                      "ctachoeiozcpkmst","sdottmimvqvfhzlk",
                      "whcqrtwarljaqocm",
                      "yqrgxthbbzmruvwy","adzzjitkyqlberpu",
                      "csxjshhnfbtgjcgm",
                      "ngksfbgkdeufmhfy","zzlzzujtugbfpsvv",
                      "swxgkelaxkoffszz","ogrmvnhwyeydwcxi",
                      "vikmjrynreazqubj",
                      "gjchrdhbeixppooh","jkarjtlhihuxqzfm",
                      "nfmbusxwwqhsaquy",
                      "nxwedpnhirijkodc","nrmzpcqkbzgmsdeo",
                      "ufpwnqycocwwbgqi",
                      "ehtbxdjhvcwdapsg","ejbxcyhffvcouoxd",
                      "iigklaveqvybkbid",
                      "qwcrrrebwyeauczj","cmjjolnwfprpzntz",
                      "gspurupoewenqznk","xiwnirovwicymtif",
                      "mcadxmmocjhzzbtt",
                      "ojribuhtopqgkqpp","bnvgzfegimthyhyo",
                      "kjdumkaiaeblbxtt",
                      "kwuuuvwdrjkyqyfv","svcvmlpsqtzbrmnz",
                      "fuddhlszptfmosir",
                      "ggaokfjtqxyctvok","dwhlbcevejvegsob",
                      "gjblfwqtnckjletn",
                      "wtxtgodhmneofvzz","ygsnfanduarpqvrn",
                      "ysonsqntnqnqagnn",
                      "yzdbjmwwtofxmpaz","ptaxsjwbissrpvdm",
                      "ivhhwynrahlruefk","pdljbgzzhxrhnqmu",
                      "vpzzxdehhwlzsgrp",
                      "rxyndewyvbophaku","taovawittfogygzi",
                      "aysnuezuqgjioyyf",
                      "ipauahivutejsrev","sioekxjbocpzrjzi",
                      "tlhipnhcbdhvhgyw",
                      "ofrkezlcbbluncri","ozmdlzfsareqmkon",
                      "qfvolfbvalczrcko",
                      "ejeggxbwhufjtjhd","urbnjdherequimyo",
                      "lmqoiaqyftqublmk","wyoxchtoecahbyjm",
                      "zfylmujpvzgqqfxo",
                      "aceqpjprqgzhffuw","iemmvtjtejhlteqa",
                      "kfurwythfncqbrxs",
                      "fpfzaadmykntrupr","stgeqvsewqntykyo",
                      "rwtwnvhjqabvovnz",
                      "yttvzqeuddvehiqu","eyrwkwxecpzxzscp",
                      "nofmcfnaiuzlqgrk",
                      "usmpkujeknoxdqrc","gzebcnjcmqioqcjb",
                      "rnrkbyojyiepdvqv",
                      "qdvjpkftaveygusd","uureltetaotxxdji",
                      "isyektlfmcpmotpl","sutdaojcvfqmjnwg",
                      "tdozuksvtvtqcykp",
                      "xqydufxhniyjunrl","yybjogamsfqljfpu",
                      "eokuiduvnrtzavmr",
                      "szvfwsizhxrbklhz","wyopladghryqlrlb",
                      "kalfshwbcuoobdwe",
                      "kzqcxkrdytalrphb","nruhduwvuytxnfvh",
                      "qdsjznqzjxlekjtp",
                      "smynsodmtrrubpqq","alrfnehgsdtsunhm",
                      "obvxygchobqafuzw","esilvarzflhfmjhh",
                      "tceovgpqjjopitor",
                      "cazrxylvhylncoze","jakvzvdollijyhwm",
                      "ohgtowaarzphsifb",
                      "ambmbeydwsdljdcc","ujwlwswdwvbpacnf",
                      "lfzbrhthlxhnmhva",
                      "cxxzogxxkmkjwqui","llkwlxfjdmrqmdgq",
                      "uclvvrkbezlvaulu",
                      "xdeuhuabvdhjipnp","hhrmdevbfqiebnum",
                      "hywzsmogbhnfcaxk",
                      "rlkrrmxxdgaxangi","djyptluftbfkxtjd",
                      "olupoctwepebdqqo","wgruytvmfzalzrtb",
                      "pselomoxubpkknqo",
                      "jepialiqqsttgcid","ajtardhciglimsdi",
                      "dpklliwcxycpfriu",
                      "pyholyswkkqjmxlj","shemwbbeliuvnvvm",
                      "tjdlkefrbysjheap",
                      "abacekzzrkhtgpcp","arfzuuojdtlgxehv",
                      "jmycebfjwrkqwsxi",
                      "rulqevsymrlwrsrz","byvoguptigfevpyy",
                      "kjhuznifzeghfdra","hcfpedolsygjsofb",
                      "lqqciehjjdtelpwa",
                      "lwrjcljtxkokvnes","rqaprgqcktgrlxnv",
                      "usjukvawgoqplrph",
                      "vnxfpyxuciadydrl","kcfhiwouwwfjqtta",
                      "ggidexivtrafqwem",
                      "zxvcbwcwoqnkxxbs","mizxbkgdiuoehddq",
                      "cgkclpnidlmetsrb",
                      "mdxtphkujabwpjeu","ixbrfaoerogqomah",
                      "pheduvdlnmrchihf",
                      "coufviypetbrtevy","otrziwxmbpndmyaa",
                      "xgvsuftfggoojbdp","fklewvbxuecmupxn",
                      "zydfvjqmmwhyfuyy",
                      "efiskxgaocgqqjvr","jlxizhsfukrheysf",
                      "mbytpqiuixyvpaab",
                      "rguedwefqmzdxowu","dgwbxitzfzbegnoc",
                      "anqyvxqouldudiww",
                      "ldxjynecsqlswvbq","nyrtstlobluggnkw",
                      "gjpgirzuabhfpkjd",
                      "qbztetcodwhfmoyg","srgqbkjrwdbikmzq",
                      "gapclpflkdsbeorm","aifsqdniwqmcuqpv",
                      "oryfrzxilushvigq",
                      "rgrpzewhrznrqrna","owxrlgxbigikfgtm",
                      "aewtdnpoiopumymt",
                      "dgwtezteqyzzylho","wsfgxnwhxftjhpxw",
                      "gsooyxmnwsucrksh",
                      "aywlgifrijfokyzu","ewkcexkqpsyfnugi",
                      "nwaavqeweeqaryzv",
                      "sguprofjftozaujc","abcepdrvvynjsufa",
                      "hwldevoubgzgbhgs",
                      "gjxmrfgnorpfspbb","iqepotyqjqeebzix",
                      "snsnxmucuccvqfvz","tgvulwtrjyegawlr",
                      "doohwubeqhbkevhr",
                      "qpcebxmotqhildhx","wxzfbqtarfurwcfw",
                      "qxtqrwxfvuenelml",
                      "uahuglbjdtacoqjt","toqaaqswchaiyhsk",
                      "xjnjmxyqemqqiejp",
                      "qvwenzdmnwecdiql","fuwhdjmdexrstmmo",
                      "eslneidrjqwzpqhd",
                      "gdaxhrlhuilhiijt","dbtkrhmbfxpkqbau",
                      "rabwrzdzwjjdhbmx","zagpnfpbwgeyeufr",
                      "clmcokjtplrbzvuh",
                      "asmpttrlkodaejic","dwsasdexwmpsmowl",
                      "uhkhghxuorryhlis",
                      "onzjhhtppsfaiacz","wehkqzwvbeonajcu",
                      "jiyzqszfywhdfsil",
                      "synvsxhrexuyxpre","dlrodwgixwmoquny",
                      "uwwrbkmjbjyxutfq",
                      "bvuzvpriwqlnbjxt","rwrevaiebpmviwqz",
                      "ehapkksqqcbofeid",
                      "nkueyjctyasmotny","ixwsqebjjdlxcqsq",
                      "htedybhazfjiueyj","xyuwuxlpirkzkqdb",
                      "epbwnmcyogpybxlm",
                      "odpuaztxnyumdvvc","saempmkfulqhwfqk",
                      "tddtoayhfpdtxokp",
                      "iknapxqudqotqiig","vudjowytbogxkrcy",
                      "pyykjiriqrhjduly",
                      "hpohizpkyzvwunni","zkreetxvsoihwkgo",
                      "rrlvhbnzrdtphqnl",
                      "dyzvyrmcdyybbddd","ammbrasbxojlitmt",
                      "dpcnodgqfivkhxvn","gujwvdfcmmqcwxfi",
                      "xpxsjmglcvcsxwdy",
                      "nhoebceeiacnmvym","dweqmfoluivgiayj",
                      "wyqgeeclrqbihfpk",
                      "guiimarisyyjqnfg","yvlkrzgjhwrlyihc",
                      "yzxgnwgpnrdprtbh",
                      "pustczakchcimwuy","zspzyfdefowgwddf",
                      "nhwgapjtnadqqaul",
                      "kpciudedjlrqsfte","ngombkqqomblyxwv",
                      "bsiyfrkwdyptmwji",
                      "jjjvjaxpzvlbryfd","jrwdpzrmxqlzzepk",
                      "wpsyqubfrhdspxkx","demgvtbzilochupd",
                      "hkazsxqvbtmawovu",
                      "fadjogsnmecatcfb","xzdsapxqliboezbc",
                      "gfhjqtkgvomiygvx",
                      "nsymgnybdjqxudvj","jgkpiuuctpywtrlh",
                      "gjbalugsikirqoam",
                      "dllcylnkzeegtsgr","pjbnwqhnqczouirt",
                      "rrqbtdjvuwwxtusj",
                      "kbgblyclstrmicux","dtpbahjtnmyuxqno",
                      "quslbttvcitxzeiy","yfalryaixpzfoihd",
                      "ettwalwfkzvwdasa",
                      "rcxmbwwsxkkkyyjs","jhdjdpthkztnjvmb",
                      "nilvygybpajtnxnr",
                      "xaklvfxsplowrglp","ubttjiaeeuwzcclq",
                      "uzmldekmvczimsrj",
                      "zoypfizhpbtpjwpv","xkzehzohmfrsmolg",
                      "iulvirmzdntweaee",
                      "kvcddisqpkysmvvo","prtnwsypyfnshpqx",
                      "aparvvfowrjncdhp",
                      "ponwkmeaxagundzq","xjaddkudsebowzen",
                      "zqruwnlzuefcpqjm","iwhqpdfuhrsxyqxe",
                      "kzwthrslljkmbqur",
                      "jjycmklnkdivnypu","svmjzfcsvgxiwwjt",
                      "hselphnqlvecmmyx",
                      "lqkdgbosdzrtitgx","tdgkjlphosocwbgu",
                      "swjkmyqytzxjwgag",
                      "biqzvbfzjivqmrro","johsjccpkithubii",
                      "rthsjeyjgdlmkygk"),
    vh_mm_pop = c(0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0126582278481013,
                  0.0126582278481013,0.0126582278481013,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0253164556962025,0.0253164556962025,
                  0.0253164556962025,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0379746835443038,
                  0.0379746835443038,
                  0.0379746835443038,0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0506329113924051,
                  0.0506329113924051,0.0506329113924051,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0632911392405063,
                  0.0632911392405063,0.0632911392405063,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0759493670886076,
                  0.0759493670886076,0.0759493670886076,
                  0.0886075949367089,
                  0.0886075949367089,0.0886075949367089,
                  0.0886075949367089,
                  0.0886075949367089,0.0886075949367089,
                  0.0886075949367089,0.0886075949367089,
                  0.0886075949367089,
                  0.0886075949367089,0.0886075949367089,
                  0.0886075949367089,
                  0.0886075949367089,0.0886075949367089,
                  0.0886075949367089,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.10126582278481,0.10126582278481,
                  0.113924050632911,
                  0.113924050632911,0.113924050632911,
                  0.113924050632911,0.113924050632911,
                  0.113924050632911,
                  0.113924050632911,0.113924050632911,
                  0.113924050632911,0.113924050632911,
                  0.113924050632911,
                  0.113924050632911,0.113924050632911,
                  0.126582278481013,
                  0.126582278481013,0.126582278481013,
                  0.126582278481013,0.126582278481013,
                  0.126582278481013,
                  0.126582278481013,0.126582278481013,
                  0.126582278481013,0.139240506329114,
                  0.139240506329114,
                  0.139240506329114,0.139240506329114,
                  0.139240506329114,
                  0.139240506329114,0.151898734177215,
                  0.151898734177215,0.151898734177215,
                  0.151898734177215,
                  0.151898734177215,0.151898734177215,
                  0.164556962025316,0.164556962025316,
                  0.164556962025316,
                  0.177215189873418,0.177215189873418,
                  0.177215189873418,0.177215189873418,
                  0.177215189873418,
                  0.177215189873418,0.177215189873418,
                  0.189873417721519,
                  0.189873417721519,0.189873417721519,
                  0.20253164556962,0.20253164556962,
                  0.20253164556962,0.20253164556962,
                  0.215189873417722,
                  0.227848101265823,0.227848101265823,
                  0.240506329113924,
                  0.240506329113924,0.253164556962025,
                  0.253164556962025,0.253164556962025,
                  0.265822784810127,
                  0.265822784810127,0.265822784810127,
                  0.265822784810127,0.265822784810127,
                  0.265822784810127,
                  0.278481012658228,0.278481012658228,
                  0.278481012658228,
                  0.291139240506329,0.291139240506329,
                  0.291139240506329,0.291139240506329,
                  0.30379746835443,
                  0.316455696202532,0.329113924050633,
                  0.329113924050633,0.329113924050633,
                  0.341772151898734,
                  0.354430379746835,0.367088607594937,
                  0.379746835443038,0.379746835443038,
                  0.379746835443038,
                  0.392405063291139,0.392405063291139,
                  0.40506329113924,0.417721518987342,
                  0.417721518987342,
                  0.430379746835443,0.430379746835443,
                  0.430379746835443,
                  0.443037974683544,0.443037974683544,
                  0.455696202531646,0.455696202531646,
                  0.468354430379747,
                  0.468354430379747,0.481012658227848,
                  0.493670886075949,0.493670886075949,
                  0.506329113924051,
                  0.518987341772152,0.531645569620253,
                  0.544303797468354,
                  0.556962025316456,0.569620253164557,
                  0.582278481012658,0.594936708860759,
                  0.594936708860759,
                  0.607594936708861,0.607594936708861,
                  0.607594936708861,0.620253164556962,
                  0.620253164556962,
                  0.632911392405063,0.645569620253165,
                  0.658227848101266,0.670886075949367,
                  0.683544303797468,
                  0.69620253164557,0.708860759493671,
                  0.721518987341772,
                  0.721518987341772,0.721518987341772,
                  0.734177215189873,0.746835443037975,
                  0.759493670886076,
                  0.772151898734177,0.784810126582278,
                  0.79746835443038,0.810126582278481,
                  0.822784810126582,
                  0.835443037974684,0.848101265822785,
                  0.860759493670886,0.873417721518987,
                  0.886075949367089,
                  0.89873417721519,0.911392405063291,
                  0.924050632911392,0.936708860759494,
                  0.949367088607595,
                  0.962025316455696,0.974683544303797,
                  0.987341772151899)
  )
  return(vhmm_pop_train)
}
vhmm_pop_train <- vh_popularity_train()


add_vh_popularity_train <- function(input){
  
  input <- left_join(input,vhmm_pop_train, by = "vh_make_model")
  
  # fix for new vh that aren't in the group list
  
  input <- input %>% mutate(vh_mm_pop = case_when(is.na(vh_mm_pop) ~ 0,
                                                  TRUE ~as.numeric(vh_mm_pop)
  )
  )
  
  #input <- input[ , !(names(input) %in% "vh_make_model")]
  
  return(input)
}

  



vh_order <- function(){
  
  vhmm_ordering <- data.frame(
    stringsAsFactors = FALSE,
    vh_make_model = c("eokuiduvnrtzavmr",
                      "lqqciehjjdtelpwa",
                      "synvsxhrexuyxpre",
                      "yvlkrzgjhwrlyihc",
                      "ldejndeewhhlcvgc",
                      "swjkmyqytzxjwgag",
                      "ujgldnoigollndkj",
                      "lqkdgbosdzrtitgx",
                      "aewtdnpoiopumymt",
                      "qxtqrwxfvuenelml",
                      "hselphnqlvecmmyx",
                      "yfalryaixpzfoihd",
                      "xdeuhuabvdhjipnp",
                      "jjycmklnkdivnypu",
                      "nsymgnybdjqxudvj",
                      "drptidaltxzxopwv",
                      "xxppvuhwmnezefxy",
                      "quslbttvcitxzeiy",
                      "byvoguptigfevpyy",
                      "cxxzogxxkmkjwqui",
                      "llkwlxfjdmrqmdgq",
                      "kbgblyclstrmicux",
                      "ptaxsjwbissrpvdm",
                      "dlrodwgixwmoquny",
                      "guiimarisyyjqnfg",
                      "asmpttrlkodaejic",
                      "zakviitdfvxsgkow",
                      "nilvygybpajtnxnr",
                      "ajtardhciglimsdi",
                      "arcsdpohuzvikyaw",
                      "mdxtphkujabwpjeu",
                      "tlhipnhcbdhvhgyw",
                      "rrlvhbnzrdtphqnl",
                      "szlkmablxrjoubla",
                      "hcfpedolsygjsofb",
                      "zspzyfdefowgwddf",
                      "dyzvyrmcdyybbddd",
                      "wehkqzwvbeonajcu",
                      "gjchrdhbeixppooh",
                      "obvxygchobqafuzw",
                      "gqfadgvnztixxbmv",
                      "ofrkezlcbbluncri",
                      "dweqmfoluivgiayj",
                      "jjjvjaxpzvlbryfd",
                      "xkzehzohmfrsmolg",
                      "usmpkujeknoxdqrc",
                      "pselomoxubpkknqo",
                      "ehapkksqqcbofeid",
                      "onzjhhtppsfaiacz",
                      "bsiyfrkwdyptmwji",
                      "cazrxylvhylncoze",
                      "blcuqlgntjavsyhs",
                      "huoicgalccftwyvz",
                      "joosvbazdbslkqgx",
                      "gfhjqtkgvomiygvx",
                      "cgrdxjyaxssrszjz",
                      "cmmuslxsfluvfyof",
                      "fuddhlszptfmosir",
                      "jrwdpzrmxqlzzepk",
                      "wxzfbqtarfurwcfw",
                      "sguprofjftozaujc",
                      "rcxmbwwsxkkkyyjs",
                      "tdozuksvtvtqcykp",
                      "dgwbxitzfzbegnoc",
                      "hhrmdevbfqiebnum",
                      "dwhlbcevejvegsob",
                      "xjsozzwcppavldee",
                      "arfzuuojdtlgxehv",
                      "xaklvfxsplowrglp",
                      "snpaaoiipfuxmvol",
                      "oryfrzxilushvigq",
                      "pdljbgzzhxrhnqmu",
                      "ctachoeiozcpkmst",
                      "zkreetxvsoihwkgo",
                      "iwhqpdfuhrsxyqxe",
                      "kalfshwbcuoobdwe",
                      "qewzxgvvhqhkfcxe",
                      "yhcelpjnbxpsmoez",
                      "ewkcexkqpsyfnugi",
                      "cqdmtwkacajclcml",
                      "tdgkjlphosocwbgu",
                      "mcadxmmocjhzzbtt",
                      "vikmjrynreazqubj",
                      "abcepdrvvynjsufa",
                      "eudwptcohxaazhpt",
                      "jmycebfjwrkqwsxi",
                      "urullqqbaabxllxl",
                      "smynsodmtrrubpqq",
                      "alrfnehgsdtsunhm",
                      "gwptulznqgygeegy",
                      "ieqgavmmxulqlvvl",
                      "vjxfnjqgvugwjhia",
                      "ettwalwfkzvwdasa",
                      "qfvolfbvalczrcko",
                      "tkqxtjbbrzagooya",
                      "xgvsuftfggoojbdp",
                      "csxjshhnfbtgjcgm",
                      "wyoxchtoecahbyjm",
                      "ukqbsscpgbfatdhs",
                      "qnixeczkijjyiprb",
                      "rxyndewyvbophaku",
                      "hkazsxqvbtmawovu",
                      "mizxbkgdiuoehddq",
                      "odpuaztxnyumdvvc",
                      "rabwrzdzwjjdhbmx",
                      "nmhahirmbvqxhxgg",
                      "zcqptmhakcmihiry",
                      "ammbrasbxojlitmt",
                      "ubttjiaeeuwzcclq",
                      "svmjzfcsvgxiwwjt",
                      "ybmkbrazyartpatx",
                      "fdbwfjqkichwdebq",
                      "jedhlhdmkdprvyex",
                      "rbxibrjokiihgfjb",
                      "zbprczwzmlxgqykc",
                      "xhczitnzxmxxebeq",
                      "ixbrfaoerogqomah",
                      "gjblfwqtnckjletn",
                      "gdaxhrlhuilhiijt",
                      "nruhduwvuytxnfvh",
                      "uureltetaotxxdji",
                      "aifsqdniwqmcuqpv",
                      "rguedwefqmzdxowu",
                      "tduddcyerrjazjsh",
                      "xjnjmxyqemqqiejp",
                      "efiskxgaocgqqjvr",
                      "clsrzyechukbaeat",
                      "jeckddxjsdolnuhe",
                      "vdwfxzfzbybhsmay",
                      "adzzjitkyqlberpu",
                      "gjpgirzuabhfpkjd",
                      "wgruytvmfzalzrtb",
                      "cqewccykrcmvawlo",
                      "rqklbykswxeuovdn",
                      "dpklliwcxycpfriu",
                      "tceovgpqjjopitor",
                      "vokpwtikxckeemdi",
                      "dlbnpwopifytzerl",
                      "vjarzxzevsdnftcl",
                      "dlrvgwmumnwcjixm",
                      "snsnxmucuccvqfvz",
                      "rgrpzewhrznrqrna",
                      "gguphuccgeqyojbl",
                      "tddtoayhfpdtxokp",
                      "ebdcmhmtqnfkaalo",
                      "rnrkbyojyiepdvqv",
                      "wsfgxnwhxftjhpxw",
                      "pheduvdlnmrchihf",
                      "mymdahqxtsywqpdn",
                      "qbkipjmisqllqwzy",
                      "kpciudedjlrqsfte",
                      "yttvzqeuddvehiqu",
                      "otrziwxmbpndmyaa",
                      "cnlvybtdupkcwczn",
                      "giyhzprslgbwsaeu",
                      "nkueyjctyasmotny",
                      "bnvgzfegimthyhyo",
                      "kjdumkaiaeblbxtt",
                      "kqxycgbergacgcei",
                      "uahuglbjdtacoqjt",
                      "cuxaapvakeemmbaa",
                      "nzxlhibmhrtafeav",
                      "fuwhdjmdexrstmmo",
                      "jsudrcgsrfddwixw",
                      "eyrwkwxecpzxzscp",
                      "anqyvxqouldudiww",
                      "nwaavqeweeqaryzv",
                      "toqaaqswchaiyhsk",
                      "aceqpjprqgzhffuw",
                      "kfurwythfncqbrxs",
                      "qpcebxmotqhildhx",
                      "sdottmimvqvfhzlk",
                      "uhkhghxuorryhlis",
                      "xzdsapxqliboezbc",
                      "iljhlfeengkciosq",
                      "xuioboiuzasnmuva",
                      "cufklbvsirnawzmv",
                      "edlxghhjgpmvhabz",
                      "ejlwzigdhipvpndt",
                      "pbrroilhklrifbwq",
                      "nolayrxwnjwzgtoo",
                      "ugwrafmvdavbsrzl",
                      "xsmqbeukcqahbfgl",
                      "ztwsplndgicacmuu",
                      "ciuxczxwhwbxdkdf",
                      "flpmjcetsinyjimc",
                      "pvrjjyumueakzstw",
                      "qzgaezfhutbcnkuf",
                      "ejbxcyhffvcouoxd",
                      "iigklaveqvybkbid",
                      "nrmzpcqkbzgmsdeo",
                      "lwrjcljtxkokvnes",
                      "ponwkmeaxagundzq",
                      "doohwubeqhbkevhr",
                      "zydfvjqmmwhyfuyy",
                      "wtxtgodhmneofvzz",
                      "saempmkfulqhwfqk",
                      "djyptluftbfkxtjd",
                      "aivacsqryguqpdib",
                      "jiyzqszfywhdfsil",
                      "nxwedpnhirijkodc",
                      "sutdaojcvfqmjnwg",
                      "jkguypwgxebmtnkx",
                      "tupmlwnkgjcgcmuv",
                      "rwtwnvhjqabvovnz",
                      "gmzbnaysqjpkzqbt",
                      "innngarflvbnwntw",
                      "kowgdytyvjhvcmta",
                      "mpmchhrcazhsvjgc",
                      "xoetbcowyxwukawr",
                      "jakvzvdollijyhwm",
                      "olupoctwepebdqqo",
                      "wyopladghryqlrlb",
                      "rqaprgqcktgrlxnv",
                      "agowcnternxraavr",
                      "dcjzdpoxqvgnjpmi",
                      "squxtuwvjnzbhzsc",
                      "uhqwwluaswbuqqjc",
                      "unmxysjyilftwsvy",
                      "ngksfbgkdeufmhfy",
                      "whcqrtwarljaqocm",
                      "lfzbrhthlxhnmhva",
                      "ygsnfanduarpqvrn",
                      "nhoebceeiacnmvym",
                      "gujwvdfcmmqcwxfi",
                      "jrwemlawxsvnwrxv",
                      "uzrhbfduaqijosql",
                      "gxgjyxrnnugizdvf",
                      "hvjwbevmcmjpnknw",
                      "kwxjejihbgmtnagf",
                      "nhqkbmwihkfvhjxx",
                      "skwelgffvlzgmbro",
                      "udlfdefgndowttah",
                      "xiwnirovwicymtif",
                      "jlxizhsfukrheysf",
                      "dsqmtbudvjtnnjwq",
                      "jcefoutonncubdss",
                      "uotpflqyvprslxjc",
                      "dqqtizjjhjmqdqqb",
                      "hcoxxbfccserxklx",
                      "kguahfjnmerrbtpp",
                      "lxvjgyjdszxtcryf",
                      "nbxjozrynlospbso",
                      "uxfxpraspeoqtmbg",
                      "ygqjgwzgeierkcpj",
                      "qvwenzdmnwecdiql",
                      "rwrevaiebpmviwqz",
                      "pustczakchcimwuy",
                      "htedybhazfjiueyj",
                      "mbytpqiuixyvpaab",
                      "xpxsjmglcvcsxwdy",
                      "wchxrbrhstsmhdsk",
                      "cnicorpxweynumqk",
                      "meratbpknllwoefn",
                      "noxmlxlzirrxdriv",
                      "vdetuihriafhetdl",
                      "wrzuftzqwoiwsmfc",
                      "zojbyremtnxajomo",
                      "yzdbjmwwtofxmpaz",
                      "vgaanttvdscmqmjr",
                      "vxumxjoeywcphfoo",
                      "xsonelzsqbpcodxe",
                      "yglorajvvrsviget",
                      "dhjmmmtnpcnalzna",
                      "rulqevsymrlwrsrz",
                      "lcokgbxbqigkqzcw",
                      "dvshwarqhxfcgwfd",
                      "gtvhxebtkefavzhg",
                      "hayciibjzwapccnb",
                      "ismjlsoibleinjdp",
                      "jixkbeuswaznqplh",
                      "kdsbtuikoaulynsu",
                      "ktrfapbareyzyyyq",
                      "zycfmwxhaaaxdwpb",
                      "ruposftqgswlcyou",
                      "htppstzpipwjtuia",
                      "ablxjgbyowxrfxed",
                      "adhoqfsfdpetomvs",
                      "ajktbllxjzfdtwpy",
                      "azxtekfvyycfmnpt",
                      "baqjsealekltnrgg",
                      "btjxvrgfduskmpts",
                      "cictcfpmfdmknnye",
                      "dohuwjuguzyvqaqg",
                      "dtdrfrtruyhvbztx",
                      "dxpafctvukcmaqao",
                      "dzbwjjmruyqxyvms",
                      "eivjhovgfnfctgjy",
                      "eyaqhofitsegmcwi",
                      "ezaffjpqpacrufvd",
                      "fkltkgzmjnzqzlqv",
                      "fnfpmchfyyqmdtfm",
                      "fvflhdedljqrcqle",
                      "gdultxlilvdnuwso",
                      "ggzcspiycgszcunf",
                      "iditakunbaxfjcmc",
                      "ifalilovsdszxmjm",
                      "jlibzlturkpyjavf",
                      "kgezpfvpmpmdicts",
                      "lpwtmtiwkgbwhufg",
                      "luwiodhzrjjobjlw",
                      "mbjevmuapzxqjnwg",
                      "mkbpzddzmalsleud",
                      "mshhupropfijhilz",
                      "mvvztpapgrcwgrdt",
                      "mzlcdmigakbbuzli",
                      "nnzwevftfeodipkn",
                      "obkqpwjualnnwgrt",
                      "okeuihmplbxhxceo",
                      "owkiszjuntmwilff",
                      "ozpyjjijxdpztngv",
                      "pmxjblqhvpwflkwt",
                      "qgnqfinpenszbzig",
                      "qidpxyunryowizua",
                      "qnesuhpxsptzihzg",
                      "qukbrubjquwstnyf",
                      "reolzfmikorzxstf",
                      "rhxboadaoyvvgflk",
                      "txgvnaysouvjtkrb",
                      "ulrbzlswbgzvmpas",
                      "unlqqlfvajjczyks",
                      "vpvtqlxqaiejzrqo",
                      "vylopbnfewdzeury",
                      "wcjrtzzkemciejsz",
                      "zzubfikjmmfsxhbn",
                      "ayeiibefzqqbyksg",
                      "cedgzkylsgxnlcjg",
                      "rrsrcesavzhbjqwk",
                      "xiesnbkcyzrpzlyq",
                      "atsglyxkfbaztzlj",
                      "gbkevbmczkqhkmoc",
                      "gdtzpvajphaxanpi",
                      "heicadwqfavetjwx",
                      "infvsqmvfzjpyfae",
                      "ioqpncqqlflrjzkj",
                      "lwwzmxipnntydwir",
                      "rzjssfxzzoddvgdc",
                      "sgknghheolfpzuid",
                      "swacqepcxnosmcll",
                      "tegzsblugaczvdmy",
                      "tmikjfqekaorgssv",
                      "tsfyxgkwdidzgzpg",
                      "ufqiiuoxasjwxqbf",
                      "vxvmxuncsxygbrzd",
                      "wrvfnbtdqgpsnzic",
                      "xewlloxrajhpbuwy",
                      "ybsenzutfrjternf",
                      "bfmdeosllvjkezwq",
                      "blmjcblhzfqwhgew",
                      "djxdgbpuyerxgrmx",
                      "dkgrgmlhhtnvzmps",
                      "exutskjkecvotaxd",
                      "fozvmjndontqoxpg",
                      "ipyrvtdugjovdwzv",
                      "mjwrreshlbmzkwmc",
                      "optzzqvbwwriedfo",
                      "pfwcfdvpkuyucnkn",
                      "tafluhgrtixdlhpv",
                      "yljpdxzmshdpmyhl",
                      "abipwhwqnzenjxfn",
                      "acvypvzmenxkevbm",
                      "admgymnmeilfhmji",
                      "aewtczgiyochvagl",
                      "afmufyguudlwbcix",
                      "aggyqhwjksgqtxdd",
                      "ajdmkzcduerbdsww",
                      "ajvhjkzguyeszaqp",
                      "akqknybjyxwbdpot",
                      "anwpfxivfvhnobvz",
                      "aoytjdcfreqvurza",
                      "asbtrxjnhqdpazot",
                      "avrwlknteymnpjpk",
                      "bawsoqdugnynetyj",
                      "bdmklueoovgkajff",
                      "bfmvfelwblrzqfyr",
                      "bgbhznmwwidntzab",
                      "bgqrpfiflzijywyu",
                      "bkwszkqrqybfgpyn",
                      "bowuhkfextvyabch",
                      "bpqxbrvavqshzebb",
                      "bpuzzsqfyvebjzjg",
                      "bqcnaxkvbmfieysy",
                      "brjgjnnpueqkyaxo",
                      "buuihjqtdgilqzjc",
                      "bvfbihgnteuiuaov",
                      "bvkytcvosbaunupg",
                      "bwjkokfezucsuigb",
                      "bxksiwcqwmxjcbci",
                      "bxrkvmsmoqvefhra",
                      "bxzfdlphpiwyjeys",
                      "bzsxlzwfqbnmljsm",
                      "caovvakxarqpgymh",
                      "ccxwaznvwtdltwlt",
                      "cedczcxvthqqkwvn",
                      "ceswaufhjtmqcndn",
                      "ckxqqcnqrqxijmmf",
                      "clcqzivttlcdfpnv",
                      "cllupxtcyclounsg",
                      "cnvpgiyrcrbsvtxo",
                      "cpixpqtyjwdgmldj",
                      "cwrigmmyfzesuezf",
                      "cwxtybsrimchiwdv",
                      "cxvltpchlhlatjkb",
                      "cyftaexytlgvmcbd",
                      "degdppvcniqrzruc",
                      "dlemjwpmokwptnai",
                      "dluodrxtjdtvbxug",
                      "dlwcludeemsmffyb",
                      "dqmsefrpxrwielmk",
                      "dqxenajfgcimjgnw",
                      "dvmnbbkcvcgwnaen",
                      "dxoirhatawazqmey",
                      "dychjlsxfaurgode",
                      "dzjyqrdmawtdcqbx",
                      "efhjvgwyjfjqsdna",
                      "efyukbppkfgttvvw",
                      "egsfpimnisvvfkne",
                      "ekwqttgkaobektch",
                      "ekztjicqomhuclqr",
                      "elvboiqxkxwhtgzg",
                      "eootycnsxmeekotz",
                      "erouvyhobhzcycuk",
                      "esiuazjovwvdlgjy",
                      "esvszhlxzbxeecme",
                      "etegzqakpcvyhkaj",
                      "evuqnfndofizyoqn",
                      "exfftzvkfnajarkm",
                      "eyrukxfjgrcdrqeo",
                      "ezjnsjxvhnocwwix",
                      "fbvdqkwltwgykywc",
                      "feioipyfbkxhcsyq",
                      "fhliexbdvrlrpjvx",
                      "fijtohsiakkeuuct",
                      "fjzqkqcjerkjykkk",
                      "fliymzbupomtmyry",
                      "flnipmkwonjnaqsp",
                      "fouvkndsdstwjqpj",
                      "frdityocokfyohoa",
                      "frvooqzltrzlbhxb",
                      "fupocenmkiiluzpe",
                      "fvrvkxucfyuyfpbk",
                      "fvsyahnxhitfllgt",
                      "fwqrokhhbukfpssj",
                      "fydlanmzkobgcfsj",
                      "fzcjreusldmxavjy",
                      "fzgvfpmdmggikezp",
                      "gfbcqaotflsogaet",
                      "gforiqpfasfwlkfl",
                      "ggqsqgrasnpkxano",
                      "gicokqmbjnafngon",
                      "gkniccewzkphqzrp",
                      "godkpvbnbdeseoct",
                      "goropquvqaoaajrk",
                      "grpzbvvgujnswyyg",
                      "gspurupoewenqznk",
                      "gvaasolsbmnbjhah",
                      "gvordmjbkxszftsl",
                      "gvsbsfrfcvftmytm",
                      "gvxirlwrjrrnoadg",
                      "gykwyopsdhbsalvd",
                      "gzpmemdiurffxomf",
                      "haowzcsrftoqsrvi",
                      "hgyoclvrybybkocm",
                      "hhidavhckwcwznhf",
                      "hikofhdgvhuwkixj",
                      "hixbnwflcimyepla",
                      "hjejiuqyfrvtxagi",
                      "hjhlpxkdgqzdlnkc",
                      "hjhvhzfpslejsnej",
                      "hotinomqpajebeov",
                      "hqixaqcgdcbagrmw",
                      "hrlyreijarvikmlk",
                      "hruelqcyvmwzsqkp",
                      "hungxfwbkelospfy",
                      "hwsgwbkydspkbben",
                      "iadmwbxpppukpjyh",
                      "iadwyxxyvkcpyeus",
                      "ibjlpnapcnsmgugu",
                      "iexbeucevqnjjbcz",
                      "ijxmcnthqquddvhc",
                      "itmcxdqtvddvmanj",
                      "itvlnddnkkmyemme",
                      "ixyvsrnksxeiqbve",
                      "iydbustazndekvfq",
                      "jancrvhjhcbxreda",
                      "jbvhqxmbarxynmfk",
                      "jcpjlgfslytgmbjq",
                      "jcxkvyjnzflnlzvh",
                      "jdsmqjpfexexznya",
                      "jghfkxkawqeujuhj",
                      "jhafhnhmasllifix",
                      "jiyhnfvmyyrpnzyx",
                      "jlhzkuikphkxcigk",
                      "jmlbcbnedxdoagqm",
                      "jskghzhjrpywrbfn",
                      "jtedxzwqoodxzcaq",
                      "jxlbmlxexeucwbue",
                      "jynbrbzntxrssxzh",
                      "kbixxyjwgxmbhcsa",
                      "kbnrpawcssaxrpmb",
                      "kbqauyzezmwspqvv",
                      "kfvusykzaeetiqtt",
                      "kglgveumqmtwrqsf",
                      "khwbllfppvhgkgzc",
                      "khzmqnkqbaqvnakh",
                      "knrylcwjpefiqlma",
                      "kpnwdujiylvsiuhp",
                      "ktpoqrjuewxmkjqr",
                      "ktytfazsvecrjvzl",
                      "kumhekfclnypkavw",
                      "kxmtwjjyzuqqgmjw",
                      "kzhhwebpekxgvfsl",
                      "kzzakxocsxhkvslf",
                      "lctcvcvytpesgryp",
                      "ldnocwfyeejbmmcy",
                      "lhamctzhosdtmdix",
                      "lhgeydlzsntbaqzj",
                      "ljwfegchielwaghb",
                      "loomciwexxewgiut",
                      "lqohoawdpvdisdiw",
                      "lqsgdewyevczcvwf",
                      "luxhsezouvtbkbpn",
                      "lvpcmycoagwxqpag",
                      "lwclhevnunilhrmm",
                      "lwhjrctubjkbhzmu",
                      "lwtlsafdbhymtibi",
                      "lxchmlyoaiocynox",
                      "lxhecyqzfsucxgqm",
                      "lybpmhaivmaqtmsq",
                      "lzsfpyidvnkaxnvs",
                      "mbnozlcufjgvpcdb",
                      "mdiqmxwkzvnpeaop",
                      "mdqyvrtwekmeflye",
                      "mmfquhvxcmjcvmhz",
                      "mpnamiwsqkvamhfa",
                      "mqsiquclpholncqd",
                      "mqzhmlqqmpafpbqw",
                      "mtcsefxrgtfdqous",
                      "mtubnuteguketfck",
                      "muixzziwtwouzapq",
                      "mxmhlvlmychxzork",
                      "mxytuavlfghapjvu",
                      "ndepxuvlaiqzdnan",
                      "nfmbusxwwqhsaquy",
                      "nfrqxttuhpuqvwti",
                      "nfxbfvlwvmxfproe",
                      "nggwrmvazdxdjyfh",
                      "ngpgrthcqiirdsux",
                      "nhembilpmgrfjifn",
                      "nhmkqmpmstaunzqh",
                      "njcwousmigzpursi",
                      "nkktflvfoasvkvht",
                      "nrwphouoeazzmbmx",
                      "ntjpzidotcatossl",
                      "nugjzznlbtmumwrp",
                      "nzgfjmknhxdezggp",
                      "obtymepcippfwigb",
                      "obzgnvzzatnjoryi",
                      "odjkyxbmtxqhkflm",
                      "oeexhaebfkkjfpff",
                      "ogdxwqtrpclsxeyw",
                      "ogyvyvhcaefqrlgk",
                      "oijipbtrzkghftpt",
                      "okzpgwvslpvgceva",
                      "opojibguvnupidif",
                      "oqbjvmfvjonftdxi",
                      "ospbwzzmmxeovscc",
                      "ouhkmefnnchsggpl",
                      "ovhdtvldyrrurawo",
                      "owkgoejsxqlzahbz",
                      "owrozlxfshxrcgvh",
                      "payritakwxpyzwqq",
                      "pbwbzedhenqmpfqt",
                      "pebdztssohmloufw",
                      "pfkfojczxwevqesz",
                      "pfvqxmrnkptcrhet",
                      "pgkgdfabkhkbviht",
                      "pijaubxodtxcsqjp",
                      "pticuqiimwdrkpdy",
                      "pvyfdiggxtjoyhqf",
                      "pyhcuhumhsoodqwl",
                      "qahlidfcpdaofkwm",
                      "qbohnomeacnwdafj",
                      "qdhfzxrzisivuhbx",
                      "qdmbicmyqrqalixj",
                      "qghhvatpvekejzpf",
                      "qicnjymlpsjmgluy",
                      "qmahqrjhkxvkwboe",
                      "qozlaoxmwusgalpz",
                      "qpjdblaqrqyuoaqk",
                      "qppmxxfbqiiallmp",
                      "qwedbcvlquqfoycc",
                      "qwqwzvbefvgugtzi",
                      "qwshkzmlvlerxsov",
                      "qxnyigoiwisibpko",
                      "qyqvfzuwfpyztbla",
                      "qzkbvcycbyxrgbqk",
                      "qzrkqxhgbqfyswsj",
                      "rclsneerlfasdcpi",
                      "rgfytoxurocumuxu",
                      "rsphcdnwdddxhdvb",
                      "rtqyfobkpliuutfx",
                      "ruyuflpnypnsgkbq",
                      "ryjiidsxttvdcpwu",
                      "rytmtyltypttvqjs",
                      "sbrarddcurfhmmqk",
                      "selnccftdsqbiurb",
                      "sghyfposeljrkedw",
                      "skgvahbwdkddoxha",
                      "smcawzwicovvejgm",
                      "sncpkctrqcditirm",
                      "spqqpwucqcaspwkb",
                      "sqqvhmadjqegpsps",
                      "suycgjdrxxvxgmha",
                      "sxmsrnbwrnvfjcvp",
                      "szluwlsqbkcnchxg",
                      "tbpblaaxsajjlyok",
                      "tcyceqtrfusfmkpy",
                      "tcyogsbbufjzekla",
                      "tdvzvrkldmrkqeth",
                      "tjdlkefrbysjheap",
                      "tkomxtfmozdiflzf",
                      "tlrnhgwgduswslyd",
                      "tlspgqlrhuzholye",
                      "tneakanblaxyevhf",
                      "tpzzxliudfwqpopv",
                      "trcsvrxdekscyvyq",
                      "trwedbipujnvnhpr",
                      "tsufyplsxyqgndsw",
                      "ttwsxzxrhwuzystf",
                      "tuudpbartgtwkoms",
                      "txcfordzmkkiicwu",
                      "tzyaldqhudfiajin",
                      "uaweumzfugnsyqmx",
                      "ubixvbvqnnjdksca",
                      "ubmafwgdsmbkfmwe",
                      "uczhoudymbvnhter",
                      "uevoqlbbbmmhkdbi",
                      "ufxhmwifrakfhfmb",
                      "ughzfvxgeziewvdi",
                      "uhhrakuolcaxvsbh",
                      "uhnrmazhzouxvsqd",
                      "ujfqhxynqnqeldes",
                      "uouhydwldpcdzuoj",
                      "updzeguaxbccwpoe",
                      "ureljkoqqvqbpdvx",
                      "uskrsfpueljrtxkg",
                      "utvgeykupnwzepks",
                      "uudtrowerqhfztjo",
                      "uudulyvocjutaxtj",
                      "uwqsodousnydlsud",
                      "uxpkyjrybttfrluy",
                      "uysllzwmzcsweunu",
                      "vafxeruawxlvrttn",
                      "vbjxuynqachujrmt",
                      "vfprhybczhnkefdf",
                      "vnsxpowqjjomnmac",
                      "vrubhbzjguaxfmlc",
                      "vtgypuzvlawmkolb",
                      "vtmylwmvnssatjlh",
                      "vuqhkgrnmheydqku",
                      "vuvbtxegkdifkviv",
                      "vwerelrvyumnkbwk",
                      "vwjjtxhpebfhzzck",
                      "vzinjyeuiebqjmep",
                      "wffzvseexopfwwjy",
                      "whfzavgfbojmgezm",
                      "wlttiymytfacsrli",
                      "wmmjbitjeevklkzj",
                      "wniogbpezwqrinyt",
                      "wnrwtzbbxbvnmbqd",
                      "wofhkqytrnqvbije",
                      "wqqxbrsnrtnuxjjl",
                      "wqsapwecaqwzorqn",
                      "wrpeamdcqpawnqag",
                      "wvnsmznngunxhcsb",
                      "wwmsynqlijbriqxy",
                      "wxcohilpavlwlnze",
                      "wykcypxicfqltavz",
                      "wylaluxuyuqkytus",
                      "xaaujpnniyiuhfql",
                      "xbdjgmfolqdfvftr",
                      "xbhnetrbyfixuzmj",
                      "xcpcxksgiefkqznu",
                      "xieosvyuphbcyzul",
                      "xjsnlswprucbsehn",
                      "xndokrsndaodfknp",
                      "xnwekpoxnvwckfcp",
                      "xokomvoaaiyuedhu",
                      "xqascjfdlnlxubce",
                      "xrmvlihddtxlbzvw",
                      "xrxxucncrqtcgixl",
                      "xsfzlqjhqbcpswcz",
                      "xuxdvcabcanlgmst",
                      "xzdakcdqrnwhtpdb",
                      "ygctwjsvugjhuylz",
                      "yhotwssicqxqetep",
                      "yhzuhoixugafkonm",
                      "ykzumnthkadrzjdb",
                      "yobesdjweimafxnq",
                      "yohngmkmrueenrvs",
                      "yqztzexmqeyeirmv",
                      "yrfmcopbrlmfinuq",
                      "yrnbnvvghdwvpayv",
                      "ysmwdnymkzsgskpv",
                      "ytckjafhdfppmrhv",
                      "ytornvfpgbsoizqr",
                      "yvuoearellwavkzs",
                      "yweystgylcxxranw",
                      "zaiwfnldcznzgrfe",
                      "zayyxxzvekzuooyq",
                      "zbqqfjqpoluazvlo",
                      "zczpxdcxdciitjcu",
                      "zeqmrfiqiusygxdd",
                      "zfesqldixdegnidi",
                      "zghfsejpgrfrqfdp",
                      "zhzmkwjsrgvudmmw",
                      "znppxsjvuytcambw",
                      "zqjvnptshpgofkqc",
                      "zrclcvscjwdbabii",
                      "zukemjnabmcizdzn",
                      "zuqrdemwihnexkpw",
                      "zzjxvhegwmgqodzk",
                      "aysnuezuqgjioyyf",
                      "qbztetcodwhfmoyg",
                      "celpzeaubkxaxxbx",
                      "cwshqcgmaazzefkx",
                      "fgxxyxcbjkodwcln",
                      "vnefzhazthgsjuax",
                      "zqswdfwtkyehitft",
                      "aawqanlavsjfqrne",
                      "ueujqvpwszzhovbj",
                      "xgikerzyofvqsmnt",
                      "kjhuznifzeghfdra",
                      "pyholyswkkqjmxlj",
                      "ngombkqqomblyxwv",
                      "bvuzvpriwqlnbjxt",
                      "rjhfsrwtoqfqvuqu",
                      "uamadghmregezetz",
                      "xebooruxiuwbpzdc",
                      "zagpnfpbwgeyeufr",
                      "dnvcqpxxzahdhbvy",
                      "suajnmrxuunoyngf",
                      "xkqlmdeookbxzzhv",
                      "xjaddkudsebowzen",
                      "xqydufxhniyjunrl",
                      "qdvjpkftaveygusd",
                      "ozmdlzfsareqmkon",
                      "dhxftxnxtxlgqcqb",
                      "ohxrgpugowiyinhv",
                      "myfrksrutuknkcnq",
                      "dgwtezteqyzzylho",
                      "exkqtrkthhgvjqdl",
                      "dqgtaigmpivatpeu",
                      "kkxluqnhrmwkfqnh",
                      "lhyhsxrxdftbsavk",
                      "xcvdausiwfrjukgn",
                      "demgvtbzilochupd",
                      "kzqcxkrdytalrphb",
                      "ejeggxbwhufjtjhd",
                      "ogrmvnhwyeydwcxi",
                      "kqubvdyyovhfxtpc",
                      "nmkzmncfytfwyfvt",
                      "tyktnjdtbrucursh",
                      "aaykjdjgdzrrdvxz",
                      "aloltvlyufzyxfvg",
                      "cfuyjykoohewxzeg",
                      "degvuccboupdnasm",
                      "disoykeofihapsal",
                      "enmicxqiumbpozpk",
                      "gtbzqhsuzzdfhzfv",
                      "kilbdkfbpczjrqek",
                      "lxjkslpwiofoynao",
                      "qjkwsppqbsgsvjwa",
                      "vomuvsgbhqzjwhgb",
                      "wkzwidzltxinpgen",
                      "wotupxkyxwcienzd",
                      "wycspyzpbmhbnmda",
                      "zwutaiivgrxnrwat",
                      "dtpbahjtnmyuxqno",
                      "vudjowytbogxkrcy",
                      "zzlzzujtugbfpsvv",
                      "ltdxvujhaocpnmzf",
                      "mcuawemlwwgaiesn",
                      "qqmkwgdqaimwcbxo",
                      "wiyyghhcxezudyxg",
                      "spbjeokdemicpdey",
                      "gctieesvmkeoozqx",
                      "qcykqtxlqnbcqfct",
                      "yxzgvihpyqafgdmy",
                      "ohgtowaarzphsifb",
                      "aywlgifrijfokyzu",
                      "dmqhptvycdmkaxbw",
                      "gdzfmtghobzpihgc",
                      "grnnfnsjjydskrht",
                      "hvziklxqbjbvncjy",
                      "ldkzuxzespcgajev",
                      "lqalilfrsznnxarm",
                      "moayoogjmiizcbez",
                      "oqkxqgmcsytmcsjz",
                      "qxksnnsrnebfkwqs",
                      "tdldeeccsirqwpcj",
                      "uzfgpmnazksmudrw",
                      "vxqfscklywhurrjp",
                      "yxovdmyzjzoutcek",
                      "gkxcvooedomgcagl",
                      "gsbeyysssgzgkkuo",
                      "ixfiagqhmszowdmf",
                      "ptbudvgjgycmmsdq",
                      "gpclrtlzecazeeev",
                      "usjukvawgoqplrph",
                      "phprbhssfhrtbeue",
                      "tksyxmdgogmokuxv",
                      "ojribuhtopqgkqpp",
                      "fszxbpjtsihsmnqv",
                      "kjogjnoblzpoxgyr",
                      "mpwepwxyokmciojj",
                      "njujuhbmnqusynwf",
                      "nwfvqtdnlrvhdbuc",
                      "qoflnrycwjlbfmow",
                      "tgfhgapnsxiewemd",
                      "ukztcjpqrpetqrnx",
                      "vqdlslwzvwucentl",
                      "abacekzzrkhtgpcp",
                      "kvcddisqpkysmvvo",
                      "cgighhnwnkxluccz",
                      "eslneidrjqwzpqhd",
                      "nofmcfnaiuzlqgrk",
                      "dohofttmidfqjozb",
                      "kmlnlefquqpparsa",
                      "mjpgppxzelxrbcnt",
                      "ssnqyyteovyaxylf",
                      "bikffjqejohkyhat",
                      "clwswcgzlaojjddv",
                      "gcmwblighdilwauf",
                      "gyhebbdhtmqwwxnp",
                      "xzwnliotgalpusga",
                      "zrfduayrhhofpqtt",
                      "oihtzffwsrwsjnfu",
                      "jkarjtlhihuxqzfm",
                      "srgqbkjrwdbikmzq",
                      "bwpieeluivljdtai",
                      "cpruzckbhhcyorgf",
                      "iwxvflrheripbuvw",
                      "qrgsdbjbjwwgirvo",
                      "xabkvgvnbqzrmnyc",
                      "ywnimisaozuyjomi",
                      "yqrgxthbbzmruvwy",
                      "gsooyxmnwsucrksh",
                      "fjimpbebyszdttpl",
                      "szduoosmrfqduakm",
                      "ttcbkakfxnfsllyq",
                      "ypgcfauffeqpeerz",
                      "jkhjcfudwqurdoex",
                      "timtcrwibllgvgxy",
                      "ehtbxdjhvcwdapsg",
                      "qwcrrrebwyeauczj",
                      "dbtkrhmbfxpkqbau",
                      "fnqgfjfkzhfbiicl",
                      "sboaeuuuhpsjujpz",
                      "uzqtsirvtxcfqnbp",
                      "vbnagxwgmwirhnjt",
                      "zedpyrfkmhzqxmaz",
                      "cjcthmigqkejxuzi",
                      "tcnfpudadgannoey",
                      "tjvewbsfsiqtqttp",
                      "tsvmlrkmftqbjvub",
                      "vnxfpyxuciadydrl",
                      "ybbdbpunwekygnto",
                      "arfkjhowhuqewzvc",
                      "qachmbxcslsazphb",
                      "tzsnmmekuhggblhv",
                      "zvdpnzgvkjkoophv",
                      "fadjogsnmecatcfb",
                      "beagnicqcxahqkeq",
                      "twhtulgwsricneea",
                      "sfcciovhmwqehacv",
                      "kcjttmlajpvbntkn",
                      "iklmkdrwatltidff",
                      "nzanewsbtbnpgrom",
                      "pqpqthiapbycbhor",
                      "hjyumbyuzbeubtbb",
                      "zsuewlbquazyrgvl",
                      "mcloznejvtelpcan",
                      "jkwlqsmedtplrvtj",
                      "drpwkafcvcypyrmw",
                      "nsgbpbjvswwlhvmm",
                      "ggaokfjtqxyctvok",
                      "ggadbhlnfgoflkaf",
                      "ifrzhyqsimoeljaa",
                      "cmjjolnwfprpzntz",
                      "iemmvtjtejhlteqa",
                      "urbnjdherequimyo",
                      "cbmmnlpqoyyursux",
                      "johsjccpkithubii",
                      "adgfkcvmsaxxghoc",
                      "ylboorftnzombypn",
                      "qdsjznqzjxlekjtp",
                      "iqepotyqjqeebzix",
                      "clmcokjtplrbzvuh",
                      "ucggoqoneixjlxxy",
                      "gxpuiivthwcmpcmc",
                      "vpvsuuudxglarezp",
                      "swxgkelaxkoffszz",
                      "owxrlgxbigikfgtm",
                      "anrwlguztftzfdng",
                      "wcyahxrmwqvhmadq",
                      "sdvssyrvwfwmdccl",
                      "kwuuuvwdrjkyqyfv",
                      "ambmbeydwsdljdcc",
                      "aocfhyagfzdywcih",
                      "pyykjiriqrhjduly",
                      "uclvvrkbezlvaulu",
                      "zfylmujpvzgqqfxo",
                      "uwwrbkmjbjyxutfq",
                      "sioekxjbocpzrjzi",
                      "ufpwnqycocwwbgqi",
                      "xoueplwxwxrzasti",
                      "ipauahivutejsrev",
                      "ivhhwynrahlruefk",
                      "ysonsqntnqnqagnn",
                      "fpfzaadmykntrupr",
                      "esilvarzflhfmjhh",
                      "taovawittfogygzi",
                      "ujwlwswdwvbpacnf",
                      "gzebcnjcmqioqcjb",
                      "svcvmlpsqtzbrmnz",
                      "uzmldekmvczimsrj",
                      "lmqoiaqyftqublmk",
                      "vpzzxdehhwlzsgrp",
                      "szvfwsizhxrbklhz",
                      "tgvulwtrjyegawlr",
                      "gjxmrfgnorpfspbb",
                      "jepialiqqsttgcid",
                      "yybjogamsfqljfpu",
                      "isyektlfmcpmotpl",
                      "hywzsmogbhnfcaxk",
                      "stgeqvsewqntykyo",
                      "fklewvbxuecmupxn",
                      "ggidexivtrafqwem",
                      "rlkrrmxxdgaxangi",
                      "kcfhiwouwwfjqtta",
                      "gapclpflkdsbeorm",
                      "shemwbbeliuvnvvm",
                      "coufviypetbrtevy",
                      "nyrtstlobluggnkw",
                      "zxvcbwcwoqnkxxbs",
                      "cgkclpnidlmetsrb",
                      "ldxjynecsqlswvbq",
                      "hwldevoubgzgbhgs",
                      "rthsjeyjgdlmkygk",
                      "iknapxqudqotqiig",
                      "ixwsqebjjdlxcqsq",
                      "xyuwuxlpirkzkqdb",
                      "dpcnodgqfivkhxvn",
                      "dwsasdexwmpsmowl",
                      "wpsyqubfrhdspxkx",
                      "gjbalugsikirqoam",
                      "epbwnmcyogpybxlm",
                      "wyqgeeclrqbihfpk",
                      "kzwthrslljkmbqur",
                      "iulvirmzdntweaee",
                      "yzxgnwgpnrdprtbh",
                      "jhdjdpthkztnjvmb",
                      "hpohizpkyzvwunni",
                      "aparvvfowrjncdhp",
                      "nhwgapjtnadqqaul",
                      "dllcylnkzeegtsgr",
                      "jgkpiuuctpywtrlh",
                      "pjbnwqhnqczouirt",
                      "zoypfizhpbtpjwpv",
                      "rrqbtdjvuwwxtusj",
                      "biqzvbfzjivqmrro",
                      "zqruwnlzuefcpqjm",
                      "prtnwsypyfnshpqx"),
    vh_mm_ord = c(363,362,361,360,
                  359,358,357,356,
                  355,354,353,352,
                  351,350,349,348,
                  347,346,345,344,
                  343,342,341,340,
                  339,338,337,336,
                  335,334,333,332,
                  331,330,329,328,
                  327,326,325,324,
                  323,322,321,320,
                  319,318,317,316,
                  315,314,313,312,
                  312,312,311,310,
                  310,309,308,307,
                  306,305,304,303,
                  302,301,300,299,
                  298,297,296,295,
                  294,293,292,291,
                  290,290,289,288,
                  287,286,285,284,
                  283,282,281,280,
                  279,278,278,278,
                  277,276,275,274,
                  273,273,272,271,
                  270,269,268,267,
                  266,265,265,264,
                  263,262,261,260,
                  260,260,260,259,
                  258,257,256,255,
                  255,254,254,253,
                  252,251,250,250,
                  249,248,247,246,
                  245,245,244,243,
                  242,241,241,240,
                  239,238,237,236,
                  235,234,233,232,
                  231,231,230,229,
                  228,227,227,226,
                  225,224,223,222,
                  221,221,220,219,
                  218,217,216,215,
                  214,214,213,212,
                  211,210,209,209,
                  208,207,207,207,
                  206,206,206,206,
                  205,205,205,205,
                  204,204,204,203,
                  202,201,200,199,
                  198,197,196,195,
                  194,193,192,192,
                  191,190,190,190,
                  190,190,189,188,
                  187,186,185,185,
                  185,185,184,183,
                  183,182,181,180,
                  179,178,178,177,
                  177,177,177,177,
                  177,176,175,174,
                  173,173,172,172,
                  172,172,172,172,
                  172,171,170,169,
                  168,167,166,165,
                  164,164,164,164,
                  164,164,163,162,
                  162,162,162,161,
                  160,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,159,
                  159,159,159,158,
                  157,156,156,156,
                  156,156,155,155,
                  155,154,154,153,
                  152,151,151,150,
                  149,148,148,148,
                  147,146,145,144,
                  143,143,142,141,
                  140,139,139,139,
                  139,138,137,136,
                  135,134,134,133,
                  132,132,132,132,
                  132,132,132,132,
                  132,132,132,132,
                  132,132,132,131,
                  130,129,128,128,
                  128,128,127,126,
                  126,126,125,124,
                  123,123,123,123,
                  123,123,123,123,
                  123,123,123,123,
                  123,122,122,122,
                  122,121,120,119,
                  119,118,117,117,
                  117,117,117,117,
                  117,117,117,116,
                  115,114,113,112,
                  111,111,111,111,
                  110,110,110,110,
                  110,110,109,108,
                  107,106,106,106,
                  106,106,106,105,
                  104,103,103,103,
                  103,102,102,101,
                  101,100,99,99,99,
                  99,99,98,98,98,
                  98,97,96,95,95,
                  95,95,94,93,93,
                  92,91,90,90,90,
                  89,89,88,87,86,
                  86,85,84,83,82,
                  81,80,79,78,77,
                  77,76,75,74,73,
                  72,71,70,69,68,
                  68,67,66,65,64,
                  63,62,61,60,59,
                  59,58,57,56,55,
                  54,53,52,51,50,
                  49,48,47,46,45,
                  44,43,42,41,40,
                  39,38,37,36,35,
                  34,33,32,31,30,
                  29,28,27,26,25,
                  24,23,22,21,20,
                  19,18,17,16,15,
                  14,13,12,11,10,
                  9,8,7,6,5,4,3,
                  2,1)
  )
  return(vhmm_ordering)
}
    vhmm_ordering <- vh_order()

add_vh_order <- function(input){
  
  input <- left_join(input,vhmm_ordering, by = "vh_make_model")
  
  # fix for new vh that aren't in the group list
  
  input <- input %>% mutate(vh_mm_ord = case_when(is.na(vh_mm_ord) ~ median(vh_mm_ord, na.rm=TRUE),
                                                          TRUE ~as.numeric(vh_mm_ord)
  )
  )
  
  #input <- input[ , !(names(input) %in% "vh_make_model")]
  
  return(input)
}



vh_popularity <- function(){
  
  vhmm_pop = data.frame(
    stringsAsFactors = FALSE,
    vh_make_model = c("adhoqfsfdpetomvs","ajktbllxjzfdtwpy",
                      "anwpfxivfvhnobvz",
                      "atsglyxkfbaztzlj","avrwlknteymnpjpk",
                      "bfmvfelwblrzqfyr",
                      "bgqrpfiflzijywyu","cedczcxvthqqkwvn",
                      "cictcfpmfdmknnye",
                      "ckxqqcnqrqxijmmf","cnvpgiyrcrbsvtxo",
                      "cwrigmmyfzesuezf","cyftaexytlgvmcbd",
                      "dlemjwpmokwptnai",
                      "dohuwjuguzyvqaqg","dqmsefrpxrwielmk",
                      "dtdrfrtruyhvbztx",
                      "efyukbppkfgttvvw","eootycnsxmeekotz",
                      "erouvyhobhzcycuk",
                      "eyrukxfjgrcdrqeo","ezaffjpqpacrufvd",
                      "fbvdqkwltwgykywc",
                      "fupocenmkiiluzpe","fvflhdedljqrcqle",
                      "gbkevbmczkqhkmoc",
                      "gdtzpvajphaxanpi","ggqsqgrasnpkxano",
                      "ggzcspiycgszcunf","gvsbsfrfcvftmytm",
                      "hayciibjzwapccnb",
                      "heicadwqfavetjwx","hikofhdgvhuwkixj",
                      "hqixaqcgdcbagrmw",
                      "hrlyreijarvikmlk","hruelqcyvmwzsqkp",
                      "htppstzpipwjtuia",
                      "hungxfwbkelospfy","infvsqmvfzjpyfae",
                      "ioqpncqqlflrjzkj",
                      "ismjlsoibleinjdp","itmcxdqtvddvmanj",
                      "jskghzhjrpywrbfn","jtedxzwqoodxzcaq",
                      "jxlbmlxexeucwbue",
                      "kbnrpawcssaxrpmb","kpnwdujiylvsiuhp",
                      "ktpoqrjuewxmkjqr",
                      "lqohoawdpvdisdiw","lqsgdewyevczcvwf",
                      "lwhjrctubjkbhzmu",
                      "lwwzmxipnntydwir","lxchmlyoaiocynox",
                      "mdiqmxwkzvnpeaop",
                      "mshhupropfijhilz","nhmkqmpmstaunzqh",
                      "obtymepcippfwigb",
                      "ogyvyvhcaefqrlgk","oijipbtrzkghftpt",
                      "ospbwzzmmxeovscc","ouhkmefnnchsggpl",
                      "pticuqiimwdrkpdy",
                      "qdhfzxrzisivuhbx","qdmbicmyqrqalixj",
                      "qyqvfzuwfpyztbla",
                      "qzrkqxhgbqfyswsj","rtqyfobkpliuutfx",
                      "rzjssfxzzoddvgdc",
                      "sgknghheolfpzuid","swacqepcxnosmcll",
                      "tbpblaaxsajjlyok",
                      "tcyogsbbufjzekla","tegzsblugaczvdmy",
                      "tkomxtfmozdiflzf","tmikjfqekaorgssv",
                      "tsfyxgkwdidzgzpg",
                      "ttwsxzxrhwuzystf","ubixvbvqnnjdksca",
                      "ufqiiuoxasjwxqbf",
                      "ughzfvxgeziewvdi","uudulyvocjutaxtj",
                      "vbjxuynqachujrmt",
                      "vfprhybczhnkefdf","vwjjtxhpebfhzzck",
                      "vxvmxuncsxygbrzd",
                      "wqsapwecaqwzorqn","wrvfnbtdqgpsnzic",
                      "wxcohilpavlwlnze",
                      "wykcypxicfqltavz","xewlloxrajhpbuwy",
                      "xokomvoaaiyuedhu","xsfzlqjhqbcpswcz",
                      "ybsenzutfrjternf",
                      "yrfmcopbrlmfinuq","ytornvfpgbsoizqr",
                      "zeqmrfiqiusygxdd",
                      "zzjxvhegwmgqodzk","ablxjgbyowxrfxed",
                      "acvypvzmenxkevbm",
                      "admgymnmeilfhmji","aewtczgiyochvagl",
                      "afmufyguudlwbcix",
                      "akqknybjyxwbdpot","aoytjdcfreqvurza",
                      "asbtrxjnhqdpazot","azxtekfvyycfmnpt",
                      "baqjsealekltnrgg",
                      "bawsoqdugnynetyj","bowuhkfextvyabch",
                      "bqcnaxkvbmfieysy",
                      "brjgjnnpueqkyaxo","btjxvrgfduskmpts",
                      "buuihjqtdgilqzjc",
                      "bvfbihgnteuiuaov","bwjkokfezucsuigb",
                      "bxksiwcqwmxjcbci",
                      "bxrkvmsmoqvefhra","bxzfdlphpiwyjeys",
                      "caovvakxarqpgymh",
                      "ccxwaznvwtdltwlt","clcqzivttlcdfpnv",
                      "cllupxtcyclounsg","cpixpqtyjwdgmldj",
                      "cxvltpchlhlatjkb",
                      "degdppvcniqrzruc","dlwcludeemsmffyb",
                      "dvshwarqhxfcgwfd",
                      "dxoirhatawazqmey","dxpafctvukcmaqao",
                      "dzbwjjmruyqxyvms",
                      "dzjyqrdmawtdcqbx","efhjvgwyjfjqsdna",
                      "egsfpimnisvvfkne",
                      "ekwqttgkaobektch","esiuazjovwvdlgjy",
                      "esvszhlxzbxeecme","exfftzvkfnajarkm",
                      "eyaqhofitsegmcwi",
                      "feioipyfbkxhcsyq","fhliexbdvrlrpjvx",
                      "fijtohsiakkeuuct",
                      "fkltkgzmjnzqzlqv","flnipmkwonjnaqsp",
                      "fnfpmchfyyqmdtfm",
                      "fouvkndsdstwjqpj","frdityocokfyohoa",
                      "frvooqzltrzlbhxb",
                      "fzcjreusldmxavjy","fzgvfpmdmggikezp",
                      "gdultxlilvdnuwso",
                      "gfbcqaotflsogaet","gicokqmbjnafngon",
                      "gtvhxebtkefavzhg","gvaasolsbmnbjhah",
                      "gykwyopsdhbsalvd",
                      "gzpmemdiurffxomf","haowzcsrftoqsrvi",
                      "hgyoclvrybybkocm",
                      "hhidavhckwcwznhf","hixbnwflcimyepla",
                      "hjejiuqyfrvtxagi",
                      "hjhlpxkdgqzdlnkc","hjhvhzfpslejsnej",
                      "hotinomqpajebeov",
                      "hwsgwbkydspkbben","iadmwbxpppukpjyh",
                      "iadwyxxyvkcpyeus","ibjlpnapcnsmgugu",
                      "iditakunbaxfjcmc",
                      "iexbeucevqnjjbcz","ifalilovsdszxmjm",
                      "ixyvsrnksxeiqbve",
                      "iydbustazndekvfq","jancrvhjhcbxreda",
                      "jbvhqxmbarxynmfk",
                      "jdsmqjpfexexznya","jghfkxkawqeujuhj",
                      "jhafhnhmasllifix",
                      "jixkbeuswaznqplh","jlibzlturkpyjavf",
                      "jynbrbzntxrssxzh",
                      "kdsbtuikoaulynsu","kfvusykzaeetiqtt",
                      "kgezpfvpmpmdicts","kglgveumqmtwrqsf",
                      "khwbllfppvhgkgzc",
                      "knrylcwjpefiqlma","ktrfapbareyzyyyq",
                      "ktytfazsvecrjvzl",
                      "kxmtwjjyzuqqgmjw","kzzakxocsxhkvslf",
                      "lcokgbxbqigkqzcw",
                      "lhamctzhosdtmdix","lhgeydlzsntbaqzj",
                      "ljwfegchielwaghb",
                      "lpwtmtiwkgbwhufg","luwiodhzrjjobjlw",
                      "lzsfpyidvnkaxnvs","mbjevmuapzxqjnwg",
                      "mdqyvrtwekmeflye",
                      "mkbpzddzmalsleud","mmfquhvxcmjcvmhz",
                      "mqsiquclpholncqd",
                      "mtcsefxrgtfdqous","mtubnuteguketfck",
                      "muixzziwtwouzapq",
                      "mvvztpapgrcwgrdt","mxmhlvlmychxzork",
                      "mxytuavlfghapjvu",
                      "mzlcdmigakbbuzli","ndepxuvlaiqzdnan",
                      "nfrqxttuhpuqvwti",
                      "nggwrmvazdxdjyfh","ngpgrthcqiirdsux",
                      "njcwousmigzpursi","nkktflvfoasvkvht",
                      "nrwphouoeazzmbmx",
                      "ntjpzidotcatossl","nugjzznlbtmumwrp",
                      "nzgfjmknhxdezggp",
                      "obkqpwjualnnwgrt","obzgnvzzatnjoryi",
                      "odjkyxbmtxqhkflm",
                      "ogdxwqtrpclsxeyw","okeuihmplbxhxceo",
                      "okzpgwvslpvgceva",
                      "oqbjvmfvjonftdxi","ovhdtvldyrrurawo",
                      "payritakwxpyzwqq","pbwbzedhenqmpfqt",
                      "pebdztssohmloufw",
                      "pfkfojczxwevqesz","pgkgdfabkhkbviht",
                      "pijaubxodtxcsqjp",
                      "pmxjblqhvpwflkwt","pyhcuhumhsoodqwl",
                      "qbohnomeacnwdafj",
                      "qicnjymlpsjmgluy","qidpxyunryowizua",
                      "qozlaoxmwusgalpz",
                      "qppmxxfbqiiallmp","qukbrubjquwstnyf",
                      "qwedbcvlquqfoycc",
                      "qwqwzvbefvgugtzi","qwshkzmlvlerxsov",
                      "qxnyigoiwisibpko","qzkbvcycbyxrgbqk",
                      "reolzfmikorzxstf",
                      "rgfytoxurocumuxu","rhxboadaoyvvgflk",
                      "rsphcdnwdddxhdvb",
                      "ryjiidsxttvdcpwu","rytmtyltypttvqjs",
                      "sncpkctrqcditirm",
                      "spqqpwucqcaspwkb","sqqvhmadjqegpsps",
                      "sxmsrnbwrnvfjcvp",
                      "tcyceqtrfusfmkpy","tdvzvrkldmrkqeth",
                      "tlrnhgwgduswslyd","tneakanblaxyevhf",
                      "trwedbipujnvnhpr",
                      "tsufyplsxyqgndsw","tuudpbartgtwkoms",
                      "txcfordzmkkiicwu",
                      "txgvnaysouvjtkrb","tzyaldqhudfiajin",
                      "uaweumzfugnsyqmx",
                      "ubmafwgdsmbkfmwe","uczhoudymbvnhter",
                      "uevoqlbbbmmhkdbi",
                      "uhhrakuolcaxvsbh","uhnrmazhzouxvsqd",
                      "ujfqhxynqnqeldes",
                      "ulrbzlswbgzvmpas","unlqqlfvajjczyks",
                      "ureljkoqqvqbpdvx","uudtrowerqhfztjo",
                      "vafxeruawxlvrttn",
                      "vnsxpowqjjomnmac","vpvtqlxqaiejzrqo",
                      "vtgypuzvlawmkolb",
                      "vtmylwmvnssatjlh","vuqhkgrnmheydqku",
                      "vuvbtxegkdifkviv",
                      "vwerelrvyumnkbwk","vylopbnfewdzeury",
                      "vzinjyeuiebqjmep",
                      "wcjrtzzkemciejsz","whfzavgfbojmgezm",
                      "wlttiymytfacsrli","wmmjbitjeevklkzj",
                      "wniogbpezwqrinyt",
                      "wnrwtzbbxbvnmbqd","wofhkqytrnqvbije",
                      "wrpeamdcqpawnqag",
                      "wvnsmznngunxhcsb","wwmsynqlijbriqxy",
                      "wylaluxuyuqkytus",
                      "xaaujpnniyiuhfql","xbdjgmfolqdfvftr",
                      "xcpcxksgiefkqznu",
                      "xieosvyuphbcyzul","xnwekpoxnvwckfcp",
                      "xqascjfdlnlxubce",
                      "xrmvlihddtxlbzvw","yhotwssicqxqetep",
                      "yhzuhoixugafkonm","yobesdjweimafxnq",
                      "yohngmkmrueenrvs",
                      "yqztzexmqeyeirmv","yrnbnvvghdwvpayv",
                      "ytckjafhdfppmrhv",
                      "yvuoearellwavkzs","zaiwfnldcznzgrfe",
                      "zfesqldixdegnidi",
                      "zghfsejpgrfrqfdp","zhzmkwjsrgvudmmw",
                      "znppxsjvuytcambw",
                      "zqjvnptshpgofkqc","zrclcvscjwdbabii",
                      "zukemjnabmcizdzn","zuqrdemwihnexkpw",
                      "zycfmwxhaaaxdwpb",
                      "zzubfikjmmfsxhbn","bwpieeluivljdtai",
                      "cnlvybtdupkcwczn",
                      "cpruzckbhhcyorgf","dluodrxtjdtvbxug",
                      "dnvcqpxxzahdhbvy",
                      "giyhzprslgbwsaeu","gkxcvooedomgcagl",
                      "gmzbnaysqjpkzqbt",
                      "gsbeyysssgzgkkuo","gvxirlwrjrrnoadg",
                      "innngarflvbnwntw",
                      "iwxvflrheripbuvw","ixfiagqhmszowdmf",
                      "kbqauyzezmwspqvv","kowgdytyvjhvcmta",
                      "loomciwexxewgiut",
                      "lxhecyqzfsucxgqm","mpmchhrcazhsvjgc",
                      "ptbudvgjgycmmsdq",
                      "qrgsdbjbjwwgirvo","suajnmrxuunoyngf",
                      "xabkvgvnbqzrmnyc",
                      "xkqlmdeookbxzzhv","xoetbcowyxwukawr",
                      "xxppvuhwmnezefxy",
                      "ywnimisaozuyjomi","bgbhznmwwidntzab",
                      "bikffjqejohkyhat","ciuxczxwhwbxdkdf",
                      "clwswcgzlaojjddv",
                      "cnicorpxweynumqk","flpmjcetsinyjimc",
                      "fvsyahnxhitfllgt",
                      "gcmwblighdilwauf","gvordmjbkxszftsl",
                      "gyhebbdhtmqwwxnp",
                      "ltdxvujhaocpnmzf","mcuawemlwwgaiesn",
                      "meratbpknllwoefn",
                      "mpnamiwsqkvamhfa","noxmlxlzirrxdriv",
                      "opojibguvnupidif",
                      "pfvqxmrnkptcrhet","pvrjjyumueakzstw",
                      "qewzxgvvhqhkfcxe","qqmkwgdqaimwcbxo",
                      "qzgaezfhutbcnkuf",
                      "rclsneerlfasdcpi","vdetuihriafhetdl",
                      "vdwfxzfzbybhsmay",
                      "wiyyghhcxezudyxg","wrzuftzqwoiwsmfc",
                      "xzwnliotgalpusga",
                      "yhcelpjnbxpsmoez","zojbyremtnxajomo",
                      "zrfduayrhhofpqtt",
                      "agowcnternxraavr","bdmklueoovgkajff",
                      "cgrdxjyaxssrszjz","cmmuslxsfluvfyof",
                      "dcjzdpoxqvgnjpmi",
                      "dqgtaigmpivatpeu","ezjnsjxvhnocwwix",
                      "fszxbpjtsihsmnqv",
                      "gwptulznqgygeegy","ieqgavmmxulqlvvl",
                      "ijxmcnthqquddvhc",
                      "jcpjlgfslytgmbjq","kjogjnoblzpoxgyr",
                      "kkxluqnhrmwkfqnh",
                      "lhyhsxrxdftbsavk","mpwepwxyokmciojj",
                      "mymdahqxtsywqpdn",
                      "njujuhbmnqusynwf","nwfvqtdnlrvhdbuc",
                      "qbkipjmisqllqwzy","qoflnrycwjlbfmow",
                      "skgvahbwdkddoxha",
                      "squxtuwvjnzbhzsc","tgfhgapnsxiewemd",
                      "uhqwwluaswbuqqjc",
                      "ukztcjpqrpetqrnx","vjxfnjqgvugwjhia",
                      "vqdlslwzvwucentl",
                      "xcvdausiwfrjukgn","ygctwjsvugjhuylz",
                      "ysmwdnymkzsgskpv",
                      "abipwhwqnzenjxfn","ajdmkzcduerbdsww",
                      "celpzeaubkxaxxbx","cwshqcgmaazzefkx",
                      "dmqhptvycdmkaxbw",
                      "edlxghhjgpmvhabz","ejlwzigdhipvpndt",
                      "elvboiqxkxwhtgzg",
                      "fdbwfjqkichwdebq","fgxxyxcbjkodwcln",
                      "gdzfmtghobzpihgc",
                      "goropquvqaoaajrk","grnnfnsjjydskrht",
                      "hvziklxqbjbvncjy",
                      "jcxkvyjnzflnlzvh","jedhlhdmkdprvyex",
                      "jiyhnfvmyyrpnzyx",
                      "kzhhwebpekxgvfsl","ldkzuxzespcgajev",
                      "lqalilfrsznnxarm","moayoogjmiizcbez",
                      "oqkxqgmcsytmcsjz",
                      "owkgoejsxqlzahbz","pbrroilhklrifbwq",
                      "qxksnnsrnebfkwqs",
                      "rbxibrjokiihgfjb","selnccftdsqbiurb",
                      "szlkmablxrjoubla",
                      "tdldeeccsirqwpcj","tpzzxliudfwqpopv",
                      "uxpkyjrybttfrluy",
                      "uzfgpmnazksmudrw","vnefzhazthgsjuax",
                      "vxqfscklywhurrjp","xuxdvcabcanlgmst",
                      "yxovdmyzjzoutcek",
                      "zbprczwzmlxgqykc","zqswdfwtkyehitft",
                      "aaykjdjgdzrrdvxz",
                      "aloltvlyufzyxfvg","bkwszkqrqybfgpyn",
                      "bpuzzsqfyvebjzjg",
                      "bzsxlzwfqbnmljsm","cfuyjykoohewxzeg",
                      "degvuccboupdnasm",
                      "disoykeofihapsal","dlbnpwopifytzerl",
                      "enmicxqiumbpozpk",
                      "etegzqakpcvyhkaj","fydlanmzkobgcfsj",
                      "gtbzqhsuzzdfhzfv","gxgjyxrnnugizdvf",
                      "hvjwbevmcmjpnknw",
                      "jlhzkuikphkxcigk","jmlbcbnedxdoagqm",
                      "kbixxyjwgxmbhcsa",
                      "khzmqnkqbaqvnakh","kilbdkfbpczjrqek",
                      "kwxjejihbgmtnagf",
                      "ldejndeewhhlcvgc","ldnocwfyeejbmmcy",
                      "lwtlsafdbhymtibi",
                      "lxjkslpwiofoynao","nhqkbmwihkfvhjxx",
                      "qahlidfcpdaofkwm","qghhvatpvekejzpf",
                      "qjkwsppqbsgsvjwa",
                      "qpjdblaqrqyuoaqk","ruyuflpnypnsgkbq",
                      "sghyfposeljrkedw",
                      "skwelgffvlzgmbro","smcawzwicovvejgm",
                      "trcsvrxdekscyvyq",
                      "udlfdefgndowttah","uouhydwldpcdzuoj",
                      "utvgeykupnwzepks",
                      "uwqsodousnydlsud","vjarzxzevsdnftcl",
                      "vomuvsgbhqzjwhgb",
                      "wkzwidzltxinpgen","wotupxkyxwcienzd",
                      "wycspyzpbmhbnmda","xjsnlswprucbsehn",
                      "zayyxxzvekzuooyq",
                      "zwutaiivgrxnrwat","aggyqhwjksgqtxdd",
                      "ajvhjkzguyeszaqp",
                      "ayeiibefzqqbyksg","bfmdeosllvjkezwq",
                      "blmjcblhzfqwhgew",
                      "bpqxbrvavqshzebb","bvkytcvosbaunupg",
                      "cedgzkylsgxnlcjg",
                      "ceswaufhjtmqcndn","cwxtybsrimchiwdv",
                      "djxdgbpuyerxgrmx","dkgrgmlhhtnvzmps",
                      "dvmnbbkcvcgwnaen",
                      "dychjlsxfaurgode","eivjhovgfnfctgjy",
                      "evuqnfndofizyoqn",
                      "exutskjkecvotaxd","fliymzbupomtmyry",
                      "fozvmjndontqoxpg",
                      "fwqrokhhbukfpssj","gforiqpfasfwlkfl",
                      "gkniccewzkphqzrp",
                      "grpzbvvgujnswyyg","ipyrvtdugjovdwzv",
                      "itvlnddnkkmyemme",
                      "kumhekfclnypkavw","lctcvcvytpesgryp",
                      "luxhsezouvtbkbpn","lvpcmycoagwxqpag",
                      "lwclhevnunilhrmm",
                      "lybpmhaivmaqtmsq","mbnozlcufjgvpcdb",
                      "mjwrreshlbmzkwmc",
                      "mqzhmlqqmpafpbqw","nfxbfvlwvmxfproe",
                      "nhembilpmgrfjifn",
                      "nnzwevftfeodipkn","optzzqvbwwriedfo",
                      "owkiszjuntmwilff",
                      "owrozlxfshxrcgvh","ozpyjjijxdpztngv",
                      "pfwcfdvpkuyucnkn","pvyfdiggxtjoyhqf",
                      "qgnqfinpenszbzig",
                      "qmahqrjhkxvkwboe","qnesuhpxsptzihzg",
                      "rrsrcesavzhbjqwk",
                      "ruposftqgswlcyou","suycgjdrxxvxgmha",
                      "szluwlsqbkcnchxg",
                      "tafluhgrtixdlhpv","tlspgqlrhuzholye",
                      "ufxhmwifrakfhfmb",
                      "updzeguaxbccwpoe","uskrsfpueljrtxkg",
                      "vrubhbzjguaxfmlc",
                      "wffzvseexopfwwjy","xiesnbkcyzrpzlyq",
                      "xndokrsndaodfknp","xzdakcdqrnwhtpdb",
                      "ykzumnthkadrzjdb",
                      "yljpdxzmshdpmyhl","yweystgylcxxranw",
                      "zbqqfjqpoluazvlo",
                      "zczpxdcxdciitjcu","adgfkcvmsaxxghoc",
                      "aivacsqryguqpdib",
                      "cgighhnwnkxluccz","iklmkdrwatltidff",
                      "kcjttmlajpvbntkn",
                      "nzanewsbtbnpgrom","pqpqthiapbycbhor",
                      "tduddcyerrjazjsh","ylboorftnzombypn",
                      "dlrvgwmumnwcjixm",
                      "fnqgfjfkzhfbiicl","gqfadgvnztixxbmv",
                      "ifrzhyqsimoeljaa",
                      "jrwemlawxsvnwrxv","nmhahirmbvqxhxgg",
                      "sboaeuuuhpsjujpz",
                      "sfcciovhmwqehacv","uysllzwmzcsweunu",
                      "uzqtsirvtxcfqnbp",
                      "uzrhbfduaqijosql","vbnagxwgmwirhnjt",
                      "xebooruxiuwbpzdc",
                      "zakviitdfvxsgkow","zcqptmhakcmihiry",
                      "zedpyrfkmhzqxmaz","mcloznejvtelpcan",
                      "oihtzffwsrwsjnfu",
                      "ujgldnoigollndkj","vgaanttvdscmqmjr",
                      "vxumxjoeywcphfoo",
                      "xrxxucncrqtcgixl","xsonelzsqbpcodxe",
                      "yglorajvvrsviget",
                      "arfkjhowhuqewzvc","beagnicqcxahqkeq",
                      "cufklbvsirnawzmv",
                      "dqxenajfgcimjgnw","jcefoutonncubdss",
                      "myfrksrutuknkcnq","phprbhssfhrtbeue",
                      "qachmbxcslsazphb",
                      "snpaaoiipfuxmvol","tksyxmdgogmokuxv",
                      "twhtulgwsricneea",
                      "tzsnmmekuhggblhv","ukqbsscpgbfatdhs",
                      "uotpflqyvprslxjc",
                      "vokpwtikxckeemdi","zvdpnzgvkjkoophv",
                      "aawqanlavsjfqrne",
                      "blcuqlgntjavsyhs","cjcthmigqkejxuzi",
                      "fjimpbebyszdttpl",
                      "gctieesvmkeoozqx","huoicgalccftwyvz",
                      "jkguypwgxebmtnkx","joosvbazdbslkqgx",
                      "kqxycgbergacgcei",
                      "qcykqtxlqnbcqfct","sbrarddcurfhmmqk",
                      "szduoosmrfqduakm",
                      "tcnfpudadgannoey","tjvewbsfsiqtqttp",
                      "tsvmlrkmftqbjvub",
                      "ttcbkakfxnfsllyq","tupmlwnkgjcgcmuv",
                      "ueujqvpwszzhovbj",
                      "xgikerzyofvqsmnt","ypgcfauffeqpeerz",
                      "yxzgvihpyqafgdmy","cqewccykrcmvawlo",
                      "dohofttmidfqjozb",
                      "dqqtizjjhjmqdqqb","drptidaltxzxopwv",
                      "ekztjicqomhuclqr",
                      "fjzqkqcjerkjykkk","godkpvbnbdeseoct",
                      "hcoxxbfccserxklx",
                      "jkhjcfudwqurdoex","kguahfjnmerrbtpp",
                      "kmlnlefquqpparsa",
                      "kqubvdyyovhfxtpc","lxvjgyjdszxtcryf",
                      "mjpgppxzelxrbcnt",
                      "nbxjozrynlospbso","nmkzmncfytfwyfvt",
                      "nolayrxwnjwzgtoo","oeexhaebfkkjfpff",
                      "rqklbykswxeuovdn",
                      "ssnqyyteovyaxylf","timtcrwibllgvgxy",
                      "ugwrafmvdavbsrzl",
                      "uxfxpraspeoqtmbg","xsmqbeukcqahbfgl",
                      "ygqjgwzgeierkcpj",
                      "ztwsplndgicacmuu","ebdcmhmtqnfkaalo",
                      "exkqtrkthhgvjqdl",
                      "ggadbhlnfgoflkaf","gpclrtlzecazeeev",
                      "sdvssyrvwfwmdccl","urullqqbaabxllxl",
                      "wchxrbrhstsmhdsk",
                      "xhczitnzxmxxebeq","anrwlguztftzfdng",
                      "cbmmnlpqoyyursux",
                      "cuxaapvakeemmbaa","nzxlhibmhrtafeav",
                      "qnixeczkijjyiprb",
                      "rjhfsrwtoqfqvuqu","spbjeokdemicpdey",
                      "uamadghmregezetz",
                      "wcyahxrmwqvhmadq","dhjmmmtnpcnalzna",
                      "drpwkafcvcypyrmw",
                      "eudwptcohxaazhpt","gguphuccgeqyojbl",
                      "nsgbpbjvswwlhvmm","tyktnjdtbrucursh",
                      "vpvsuuudxglarezp",
                      "ybmkbrazyartpatx","arcsdpohuzvikyaw",
                      "clsrzyechukbaeat",
                      "jeckddxjsdolnuhe","jkwlqsmedtplrvtj",
                      "jsudrcgsrfddwixw",
                      "tkqxtjbbrzagooya","ucggoqoneixjlxxy",
                      "ybbdbpunwekygnto",
                      "cqdmtwkacajclcml","fvrvkxucfyuyfpbk",
                      "gxpuiivthwcmpcmc","hjyumbyuzbeubtbb",
                      "iljhlfeengkciosq",
                      "unmxysjyilftwsvy","xbhnetrbyfixuzmj",
                      "xuioboiuzasnmuva",
                      "zsuewlbquazyrgvl","aceqpjprqgzhffuw",
                      "dsqmtbudvjtnnjwq",
                      "kfurwythfncqbrxs","nfmbusxwwqhsaquy",
                      "qfvolfbvalczrcko",
                      "sioekxjbocpzrjzi","taovawittfogygzi",
                      "ufpwnqycocwwbgqi",
                      "adzzjitkyqlberpu","gspurupoewenqznk",
                      "vikmjrynreazqubj","gjchrdhbeixppooh",
                      "jkarjtlhihuxqzfm",
                      "nxwedpnhirijkodc","sdottmimvqvfhzlk",
                      "zzlzzujtugbfpsvv",
                      "bnvgzfegimthyhyo","ejbxcyhffvcouoxd",
                      "iigklaveqvybkbid",
                      "mcadxmmocjhzzbtt","ngksfbgkdeufmhfy",
                      "nrmzpcqkbzgmsdeo",
                      "ogrmvnhwyeydwcxi","ojribuhtopqgkqpp",
                      "swxgkelaxkoffszz","whcqrtwarljaqocm",
                      "aocfhyagfzdywcih",
                      "cmjjolnwfprpzntz","dhxftxnxtxlgqcqb",
                      "ehtbxdjhvcwdapsg",
                      "ohxrgpugowiyinhv","qwcrrrebwyeauczj",
                      "kjdumkaiaeblbxtt",
                      "kwuuuvwdrjkyqyfv","wqqxbrsnrtnuxjjl",
                      "xiwnirovwicymtif",
                      "ggaokfjtqxyctvok","nruhduwvuytxnfvh",
                      "obvxygchobqafuzw",
                      "qdvjpkftaveygusd","rxyndewyvbophaku",
                      "tlhipnhcbdhvhgyw","uureltetaotxxdji",
                      "wyopladghryqlrlb",
                      "smynsodmtrrubpqq","vpzzxdehhwlzsgrp",
                      "fuddhlszptfmosir",
                      "gjblfwqtnckjletn","lmqoiaqyftqublmk",
                      "ygsnfanduarpqvrn",
                      "ysonsqntnqnqagnn","yzdbjmwwtofxmpaz",
                      "ctachoeiozcpkmst",
                      "ivhhwynrahlruefk","ofrkezlcbbluncri",
                      "ptaxsjwbissrpvdm","svcvmlpsqtzbrmnz",
                      "aysnuezuqgjioyyf",
                      "csxjshhnfbtgjcgm","ejeggxbwhufjtjhd",
                      "ipauahivutejsrev",
                      "urbnjdherequimyo","wtxtgodhmneofvzz",
                      "wyoxchtoecahbyjm",
                      "xjsozzwcppavldee","zfylmujpvzgqqfxo",
                      "fpfzaadmykntrupr",
                      "iemmvtjtejhlteqa","ozmdlzfsareqmkon",
                      "pdljbgzzhxrhnqmu",
                      "xoueplwxwxrzasti","yqrgxthbbzmruvwy",
                      "cxxzogxxkmkjwqui","hhrmdevbfqiebnum",
                      "stgeqvsewqntykyo",
                      "sutdaojcvfqmjnwg","xdeuhuabvdhjipnp",
                      "lfzbrhthlxhnmhva",
                      "nofmcfnaiuzlqgrk","xqydufxhniyjunrl",
                      "eyrwkwxecpzxzscp",
                      "gzebcnjcmqioqcjb","rwtwnvhjqabvovnz",
                      "tdozuksvtvtqcykp",
                      "uclvvrkbezlvaulu","yybjogamsfqljfpu",
                      "eokuiduvnrtzavmr","isyektlfmcpmotpl",
                      "kalfshwbcuoobdwe",
                      "qdsjznqzjxlekjtp","szvfwsizhxrbklhz",
                      "yttvzqeuddvehiqu",
                      "alrfnehgsdtsunhm","dwhlbcevejvegsob",
                      "kzqcxkrdytalrphb",
                      "esilvarzflhfmjhh","ohgtowaarzphsifb",
                      "kcfhiwouwwfjqtta",
                      "ambmbeydwsdljdcc","jakvzvdollijyhwm",
                      "lwrjcljtxkokvnes",
                      "rnrkbyojyiepdvqv","ujwlwswdwvbpacnf",
                      "usmpkujeknoxdqrc","cazrxylvhylncoze",
                      "hywzsmogbhnfcaxk",
                      "llkwlxfjdmrqmdgq","rlkrrmxxdgaxangi",
                      "ajtardhciglimsdi",
                      "coufviypetbrtevy","jepialiqqsttgcid",
                      "dpklliwcxycpfriu",
                      "kjhuznifzeghfdra","pselomoxubpkknqo",
                      "pyholyswkkqjmxlj",
                      "zydfvjqmmwhyfuyy","abacekzzrkhtgpcp",
                      "hcfpedolsygjsofb","jmycebfjwrkqwsxi",
                      "rqaprgqcktgrlxnv",
                      "tjdlkefrbysjheap","byvoguptigfevpyy",
                      "rulqevsymrlwrsrz",
                      "arfzuuojdtlgxehv","lqqciehjjdtelpwa",
                      "tceovgpqjjopitor",
                      "usjukvawgoqplrph","vnxfpyxuciadydrl",
                      "fklewvbxuecmupxn",
                      "cgkclpnidlmetsrb","ggidexivtrafqwem",
                      "ixbrfaoerogqomah",
                      "mizxbkgdiuoehddq","wgruytvmfzalzrtb",
                      "efiskxgaocgqqjvr","olupoctwepebdqqo",
                      "zxvcbwcwoqnkxxbs",
                      "pheduvdlnmrchihf","djyptluftbfkxtjd",
                      "mdxtphkujabwpjeu",
                      "shemwbbeliuvnvvm","xgvsuftfggoojbdp",
                      "otrziwxmbpndmyaa",
                      "oryfrzxilushvigq","qbztetcodwhfmoyg",
                      "nyrtstlobluggnkw",
                      "owxrlgxbigikfgtm","anqyvxqouldudiww",
                      "gjpgirzuabhfpkjd","ldxjynecsqlswvbq",
                      "dgwbxitzfzbegnoc",
                      "wsfgxnwhxftjhpxw","aywlgifrijfokyzu",
                      "ewkcexkqpsyfnugi",
                      "gapclpflkdsbeorm","jlxizhsfukrheysf",
                      "srgqbkjrwdbikmzq",
                      "aifsqdniwqmcuqpv","mbytpqiuixyvpaab",
                      "rguedwefqmzdxowu",
                      "aewtdnpoiopumymt","rgrpzewhrznrqrna",
                      "gsooyxmnwsucrksh",
                      "hwldevoubgzgbhgs","nwaavqeweeqaryzv",
                      "abcepdrvvynjsufa","dgwtezteqyzzylho",
                      "sguprofjftozaujc",
                      "snsnxmucuccvqfvz","tgvulwtrjyegawlr",
                      "doohwubeqhbkevhr",
                      "gjxmrfgnorpfspbb","iqepotyqjqeebzix",
                      "qpcebxmotqhildhx",
                      "qxtqrwxfvuenelml","wxzfbqtarfurwcfw",
                      "toqaaqswchaiyhsk",
                      "qvwenzdmnwecdiql","uahuglbjdtacoqjt",
                      "eslneidrjqwzpqhd","fuwhdjmdexrstmmo",
                      "xjnjmxyqemqqiejp",
                      "rabwrzdzwjjdhbmx","gdaxhrlhuilhiijt",
                      "dbtkrhmbfxpkqbau",
                      "clmcokjtplrbzvuh","uhkhghxuorryhlis",
                      "zagpnfpbwgeyeufr",
                      "onzjhhtppsfaiacz","wehkqzwvbeonajcu",
                      "asmpttrlkodaejic",
                      "dwsasdexwmpsmowl","jiyzqszfywhdfsil",
                      "dlrodwgixwmoquny",
                      "synvsxhrexuyxpre","htedybhazfjiueyj",
                      "uwwrbkmjbjyxutfq","rwrevaiebpmviwqz",
                      "bvuzvpriwqlnbjxt",
                      "ehapkksqqcbofeid","ixwsqebjjdlxcqsq",
                      "nkueyjctyasmotny",
                      "xyuwuxlpirkzkqdb","tddtoayhfpdtxokp",
                      "epbwnmcyogpybxlm",
                      "saempmkfulqhwfqk","odpuaztxnyumdvvc",
                      "iknapxqudqotqiig",
                      "pyykjiriqrhjduly","vudjowytbogxkrcy",
                      "rrlvhbnzrdtphqnl","ammbrasbxojlitmt",
                      "hpohizpkyzvwunni",
                      "zkreetxvsoihwkgo","xpxsjmglcvcsxwdy",
                      "dyzvyrmcdyybbddd",
                      "gujwvdfcmmqcwxfi","dpcnodgqfivkhxvn",
                      "nhoebceeiacnmvym",
                      "yvlkrzgjhwrlyihc","dweqmfoluivgiayj",
                      "wyqgeeclrqbihfpk",
                      "guiimarisyyjqnfg","pustczakchcimwuy",
                      "yzxgnwgpnrdprtbh",
                      "nhwgapjtnadqqaul","zspzyfdefowgwddf",
                      "bsiyfrkwdyptmwji","kpciudedjlrqsfte",
                      "ngombkqqomblyxwv",
                      "jjjvjaxpzvlbryfd","jrwdpzrmxqlzzepk",
                      "demgvtbzilochupd",
                      "wpsyqubfrhdspxkx","xzdsapxqliboezbc",
                      "hkazsxqvbtmawovu",
                      "gfhjqtkgvomiygvx","fadjogsnmecatcfb",
                      "nsymgnybdjqxudvj",
                      "gjbalugsikirqoam","jgkpiuuctpywtrlh",
                      "pjbnwqhnqczouirt","rrqbtdjvuwwxtusj",
                      "dllcylnkzeegtsgr",
                      "kbgblyclstrmicux","dtpbahjtnmyuxqno",
                      "quslbttvcitxzeiy",
                      "yfalryaixpzfoihd","rcxmbwwsxkkkyyjs",
                      "ettwalwfkzvwdasa",
                      "jhdjdpthkztnjvmb","xaklvfxsplowrglp",
                      "nilvygybpajtnxnr",
                      "ubttjiaeeuwzcclq","uzmldekmvczimsrj",
                      "zoypfizhpbtpjwpv",
                      "xkzehzohmfrsmolg","iulvirmzdntweaee",
                      "prtnwsypyfnshpqx","kvcddisqpkysmvvo",
                      "aparvvfowrjncdhp",
                      "xjaddkudsebowzen","ponwkmeaxagundzq",
                      "zqruwnlzuefcpqjm",
                      "iwhqpdfuhrsxyqxe","kzwthrslljkmbqur",
                      "jjycmklnkdivnypu",
                      "svmjzfcsvgxiwwjt","hselphnqlvecmmyx",
                      "lqkdgbosdzrtitgx",
                      "tdgkjlphosocwbgu","swjkmyqytzxjwgag",
                      "biqzvbfzjivqmrro","johsjccpkithubii",
                      "rthsjeyjgdlmkygk"),
    vh_mm_pop = c(1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,1,
                  1,1,1,1,1,1,1,1,1,1,
                  1,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,
                  2,2,2,2,2,2,2,2,2,2,2,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,
                  3,3,3,3,3,3,3,3,3,3,3,
                  3,3,3,4,4,4,4,4,4,4,
                  4,4,4,4,4,4,4,4,4,4,
                  4,4,4,4,4,4,4,4,4,4,4,
                  4,4,4,4,4,4,4,4,4,4,
                  4,4,4,4,5,5,5,5,5,5,
                  5,5,5,5,5,5,5,5,5,5,5,
                  5,5,5,5,5,5,5,5,5,5,
                  5,5,5,5,5,5,5,5,5,6,
                  6,6,6,6,6,6,6,6,6,6,6,
                  6,6,6,6,6,6,6,6,6,6,
                  6,6,6,6,6,6,6,6,6,6,
                  6,6,6,6,7,7,7,7,7,7,7,
                  7,7,7,7,7,7,7,7,7,7,
                  7,7,7,7,7,7,8,8,8,8,
                  8,8,8,8,8,8,8,8,8,9,9,
                  9,9,9,9,9,9,9,9,9,9,
                  9,9,9,9,9,9,9,9,10,10,
                  10,10,10,10,10,10,10,
                  10,10,10,10,10,10,11,11,
                  11,11,11,11,11,11,12,12,
                  12,12,12,12,12,12,12,13,
                  13,13,14,14,14,14,15,
                  15,15,15,15,15,15,16,16,
                  16,16,16,17,17,17,18,18,
                  19,20,20,20,21,21,21,21,
                  21,21,21,22,22,22,22,
                  23,23,23,24,25,25,25,26,
                  27,28,29,30,31,31,32,32,
                  33,33,33,34,35,35,35,36,
                  37,38,39,40,41,41,42,
                  43,44,45,46,47,47,48,48,
                  48,49,49,50,51,51,52,53,
                  54,55,56,56,57,58,58,59,
                  60,61,62,63,64,64,65,
                  66,67,68,69,69,70,71,72,
                  73,74,75,76,77,78)
  )
  return(vhmm_pop)
}
    vhmm_pop <- vh_popularity()

add_vh_popularity <- function(input){
  
  input <- left_join(input,vhmm_pop, by = "vh_make_model")
  
  # fix for new vh that aren't in the group list
  
  input <- input %>% mutate(vh_mm_pop = case_when(is.na(vh_mm_pop) ~ 1,
                                                  TRUE ~as.numeric(vh_mm_pop)
  )
  )
  
  #input <- input[ , !(names(input) %in% "vh_make_model")]
  
  return(input)
}



vh_weight_fix <- function(){
  vh_weight_map <- tribble(
    ~vh_make_model,~vh_weight_imp,
    "acvypvzmenxkevbm",1751,
    "adzzjitkyqlberpu",828,
    "aggyqhwjksgqtxdd",720,
    "aocfhyagfzdywcih",560,
    "arfkjhowhuqewzvc",993,
    "bawsoqdugnynetyj",720,
    "bgbhznmwwidntzab",850,
    "bwjkokfezucsuigb",801,
    "cjcthmigqkejxuzi",67,
    "clwswcgzlaojjddv",1252,
    "cnicorpxweynumqk",1243,
    "cnvpgiyrcrbsvtxo",900,
    "cpixpqtyjwdgmldj",802,
    "demgvtbzilochupd",880,
    "dohofttmidfqjozb",1721,
    "fjimpbebyszdttpl",1214,
    "fnqgfjfkzhfbiicl",940,
    "fzcjreusldmxavjy",1177,
    "gdaxhrlhuilhiijt",1496,
    "ggadbhlnfgoflkaf",859,
    "ggidexivtrafqwem",915,
    "gvxirlwrjrrnoadg",1144,
    "haowzcsrftoqsrvi",1395,
    "iadmwbxpppukpjyh",969,
    "iknapxqudqotqiig",343,
    "ipauahivutejsrev",610,
    "ipyrvtdugjovdwzv",1255,
    "ixfiagqhmszowdmf",1015,
    "ixwsqebjjdlxcqsq",85,
    "jkwlqsmedtplrvtj",1015,
    "jlibzlturkpyjavf",700,
    "jmlbcbnedxdoagqm",1130,
    "kfvusykzaeetiqtt",1252.5,
    "khwbllfppvhgkgzc",700,
    "kjogjnoblzpoxgyr",790,
    "mcuawemlwwgaiesn",725,
    "mqzhmlqqmpafpbqw",370,
    "mzlcdmigakbbuzli",793,
    "nmkzmncfytfwyfvt",809,
    "odpuaztxnyumdvvc",1014,
    "ogyvyvhcaefqrlgk",1343,
    "opojibguvnupidif",985,
    "oqkxqgmcsytmcsjz",955,
    "ozmdlzfsareqmkon",650,
    "pebdztssohmloufw",1552.5,
    "pustczakchcimwuy",770,
    "pyhcuhumhsoodqwl",801,
    "qghhvatpvekejzpf",85,
    "qicnjymlpsjmgluy",821,
    "qrgsdbjbjwwgirvo",1117,
    "rlkrrmxxdgaxangi",725,
    "rnrkbyojyiepdvqv",1852,
    "rqaprgqcktgrlxnv",1770,
    "rrqbtdjvuwwxtusj",808,
    "sioekxjbocpzrjzi",969,
    "stgeqvsewqntykyo",1060,
    "taovawittfogygzi",828,
    "tddtoayhfpdtxokp",1260,
    "tgvulwtrjyegawlr",374,
    "tjdlkefrbysjheap",343,
    "tkomxtfmozdiflzf",905,
    "uouhydwldpcdzuoj",1030,
    "vudjowytbogxkrcy",1131,
    "wofhkqytrnqvbije",790,
    "wvnsmznngunxhcsb",968,
    "wwmsynqlijbriqxy",1144,
    "xokomvoaaiyuedhu",370,
    "xuxdvcabcanlgmst",370,
    "ygctwjsvugjhuylz",775,
    "yxzgvihpyqafgdmy",830,
    "zbqqfjqpoluazvlo",600,
    "zqjvnptshpgofkqc",992
  )
  return(vh_weight_map)
}
  vh_weight_map <- vh_weight_fix()

add_vh_weight <- function(input){

  #str(vh_weight_map)
  input <- left_join(input,vh_weight_map, by = "vh_make_model")
  
  # I believe some of the weights at ZERO should actually be NAs.
  # Transforming 0 into NAs
  # And adding fixed weights to those NA for which I believe I have a better value
  #str(input)
  input <- input %>% mutate(vh_weight = ifelse(vh_weight == 0, NA, vh_weight))
  input <- input %>% mutate(vh_weight = ifelse(is.na(vh_weight), vh_weight_imp,vh_weight))
  

  
  # if there are still NAs, I'm micing it....!
  # this operation was done in the training phase, using 60k policies
  
      # input_trim <- input %>% select(vh_weight, vh_value, vh_speed)  
      # init = mice(input_trim, maxit=0) 
      # meth = init$method
      # predM = init$predictorMatrix
      # 
      # meth[c("vh_weight")]="pmm"
      # 
      # set.seed(123)
      # imputed = mice(input_trim, method=meth, predictorMatrix=predM, m=5)
      # 
      # imputed <- complete(imputed)
      # sapply(imputed, function(x) sum(is.na(x)))
      # str(imputed)
      # 
      # input$vh_weight_imp2 = imputed$vh_weight
      # 
      # input <- input %>% mutate(vh_weight = ifelse(is.na(vh_weight), vh_weight_imp2, vh_weight ))
      # 
      # summary(xy$vh_weight_imp)
      # summary(xy$vh_weight)
      # input <- input[ , !(names(input) %in% c("vh_weight_imp","vh_weight_imp2"))] 
  
  
  # With this new "complete" data, I have fitted a polynomial to have an estimate on weights
  # This will be my safety net to impute weights on missing values, if there are any left.

  
  input$vh_weight_manual_pred = -1.671374e+03 + 2.481365e+01*(input$vh_speed) + -2.586039e-02*(input$vh_speed)^2 + 
    -3.925560e-04*(input$vh_speed)^3 + 9.843998e-07*(input$vh_speed)^4 +
    2.596756e-02*(input$vh_value) + 3.181435e-07*(input$vh_value)^2 + 
    -1.192695e-11*(input$vh_value)^3 + 7.269505e-17*(input$vh_value)^4
  
  
  input <- input %>% mutate(vh_weight = ifelse(is.na(vh_weight), vh_weight_manual_pred,vh_weight))
  input <- input[ , !(names(input) %in% c("vh_weight_imp","vh_weight_imp2","vh_weight_manual_pred"))] 
  
  return(input)
}


driver_averages_fct <- function(){
  
  # I have computed various averages for each Driver Ages.
  # We will use those as a metrics on how far from the "average" a specific client is.
  # I did fit a smooth curve (loess) to get the best proxy of the general population
  # I used package datapasta to convert into code.
  
  driver_averages <- tibble::tribble(
    ~drv_age1,           ~avg_ncd,         ~avg_pol,      ~avg_polsit,         ~avg_pco,      ~avg_vh_age,    ~avg_vh_value, ~avg_vh_valuedep,         ~avg_pop,          ~avg_nyl,
    16,  0.612722430569237, 1.74758727336985, 1.72128679264027, 2.96865255501514, 9.79483441728553, 12148.2017401251, 4145.96356288294, 528.248989566953, 0.904250750853497,
    17,  0.612722430569237, 1.74758727336985, 1.72128679264027, 2.96865255501514, 9.79483441728553, 12148.2017401251, 4145.96356288294, 528.248989566953, 0.904250750853497,
    18,  0.612722430569237, 1.74758727336985, 1.72128679264027, 2.96865255501514, 9.79483441728553, 12148.2017401251, 4145.96356288294, 528.248989566953, 0.904250750853497,
    19,  0.612722430569237, 1.74758727336985, 1.72128679264027, 2.96865255501514, 9.79483441728553, 12148.2017401251, 4145.96356288294, 528.248989566953, 0.904250750853497,
    20,  0.577097318015375, 2.14898571387772, 1.91794406506457, 2.99737897569543, 9.98864755478171, 12620.8713698548, 4216.93617259724, 531.729893208573,  1.67087476495845,
    21,  0.541976701078778, 2.53968780746592,  2.1036054528151, 3.02604390999623, 10.1679840697803,  13072.310493532, 4290.50025426734, 534.834169048284,  2.46185325771688,
    22,  0.507079231108485, 2.92780298564576, 2.28989990093285, 3.05502553531357,  10.340379447376, 13514.3826905773, 4364.13742875025, 537.777101223038,  3.27743793191969,
    23,  0.472319100338071, 3.31119567327203,  2.4679822812365, 3.08390320171878, 10.4981249089335, 13948.0710113635, 4439.06542099347, 540.616037017623,  4.11424756031148,
    24,  0.438034041530677, 3.69098194868571, 2.63764027465697, 3.11237863760231,  10.648120775351, 14382.8407065373, 4515.46342547576, 543.279452550307,  4.97509523945661,
    25,   0.40353459024184, 4.07857178510291, 2.78981704728163, 3.14083066345647, 10.7421699315468, 14836.0025567593, 4590.18337290972, 546.232257972357,  5.85783804846744,
    26,  0.369586237716787, 4.43468507659788, 2.90046315450777, 3.16883266278199, 10.7531712210296,  15225.243689437, 4673.49000369515, 548.542280914433,  6.75942339110806,
    27,   0.33481833403081, 4.77085807230005, 2.99038758560239,  3.1959771888338, 10.7043122199948, 15572.6615818919, 4763.77953973675, 551.332363325683,  7.69218401853195,
    28,  0.300562507532131, 5.11146650983017, 3.05823571638897, 3.22064475719294, 10.6413149621205, 15902.1829274307, 4870.71577855715, 553.660858302599,  8.63192116204017,
    29,  0.267593416322389, 5.45824487008924,  3.1212845119185, 3.24391522817144, 10.5666677195594, 16196.8685875571, 4991.04517056782, 554.634809464306,  9.54289685019207,
    30,  0.235582984234157, 5.81901282502436, 3.18893342519034,  3.2663256513657, 10.5057895351627, 16483.3618965916, 5104.19090251812, 555.633975752075,  10.4433517332757,
    31,  0.207696018518446, 6.18287728801594, 3.25101326382532, 3.28481713769215, 10.4463816813937, 16747.4675151439, 5211.33418274148, 556.038524500807,  11.3564038678694,
    32,  0.183124427268254, 6.51294522606279, 3.31913567130736, 3.29937817335005,  10.403761972288, 16980.3217455384, 5311.94742015374, 556.211381439715,  12.2912164451492,
    33,  0.161270961924037, 6.81445802412073, 3.37902146669153, 3.31112549963475, 10.3666705518278, 17177.9961494384, 5388.91415169399, 556.398690457361,  13.2362764454598,
    34,   0.14147789689344, 7.13140373729669,  3.4329692915609, 3.32078811229142, 10.3394894354675, 17358.8019845941, 5450.80800330625, 556.293322351049,  14.2126233069281,
    35,   0.12257371632107, 7.45899363267946, 3.46531524922332, 3.32896961468898, 10.3271050103426, 17531.5733659535, 5513.49008321378, 556.305662759254,  15.2103189754152,
    36,  0.107268807084576, 7.77637106855772,  3.4931455342544,  3.3341154118237, 10.3290986268206, 17675.3021289328, 5564.18951520382,  556.71697851765,  16.1750009528527,
    37, 0.0937532095950484, 8.09117110791277,  3.5203411747231, 3.33739796212963, 10.3564366742314, 17804.5032312952, 5600.38854027302, 557.289524956022,  17.0684722696248,
    38, 0.0821176600533453, 8.43367183505355, 3.56821422775504, 3.33914416053351, 10.4042014275703, 17944.2593239141, 5637.77225196705, 557.984889298232,  17.9445234538679,
    39, 0.0736454949590628, 8.76866469887532, 3.62789548170552, 3.33915485359019, 10.4694200673767, 18083.0468774565,  5677.5387607401,  559.01949917335,  18.8130097498641,
    40, 0.0660293121085782, 9.06866990010207, 3.69645225319911, 3.33914352577984,  10.547474007642, 18216.9269261804,  5710.5355920196, 560.227279393316,  19.6628821728678,
    41, 0.0594444340093628, 9.35838845675707,  3.7624873989994,  3.3391561937257, 10.6238546147509, 18352.1196905537, 5735.77027087601, 561.614740409213,  20.5464031278985,
    42, 0.0533943922559758, 9.66302424423666, 3.81501186583387, 3.33954448865374, 10.6949524677471, 18469.3142885675, 5750.49029764783, 563.039631350362,  21.5028756260466,
    43,  0.048040219437664,  9.9357309537314, 3.84574911973736, 3.34008708693175, 10.7500864587689, 18551.9449464642, 5748.07867945884, 564.506407088606,  22.4780121231861,
    44, 0.0435346278268745, 10.1822767877377,  3.8672800198584, 3.34050540250894, 10.8013370780035, 18612.2695765806, 5731.62946004773, 565.979295050906,  23.4469446453013,
    45, 0.0394600546076326, 10.4535285431866, 3.88601443763152, 3.34132601950897, 10.8496680087185, 18649.8842104536, 5708.56518504413, 567.359178823485,  24.4261781716511,
    46, 0.0361226611468077, 10.7337598534446, 3.90457105896892, 3.34294327752945, 10.8914917312994, 18656.1339097288, 5678.58939253288, 568.777611710099,  25.3896487043433,
    47, 0.0335810160395874, 10.9737631790689, 3.92212657654287, 3.34531278217643, 10.9308004977541, 18643.8065352686,  5653.0343680723, 570.103060133516,  26.3093162458867,
    48, 0.0314031719476042, 11.2103407006099, 3.93775316268873, 3.34780669612211, 10.9591116362256, 18621.5444539478, 5632.16145670122, 571.234101857742,   27.227547656053,
    49, 0.0293308691044225, 11.4747405342986, 3.95510497263029, 3.34971203034987, 10.9808242199379, 18588.2730938718, 5608.68319663383, 572.109181245276,  28.1931060141072,
    50, 0.0275126656980134, 11.7125739107489, 3.96378737302278, 3.35095648313318, 10.9854738210175, 18553.9188981976, 5589.60784190368, 572.527787142598,   29.177363327755,
    51, 0.0261291761265233, 11.9086737148924, 3.97119354373152, 3.35202026060416, 10.9802436139142, 18520.1102101667, 5579.47609354541, 572.458206591298,  30.1437792776595,
    52, 0.0248255686404372, 12.1098114355512, 3.97724959099103, 3.35355447551537, 10.9706798883952, 18485.8654599414, 5569.49122512575, 571.975142654072,  31.1237161381889,
    53, 0.0236826967210511, 12.3259595078573, 3.99555560406218, 3.35485689896275, 10.9708734265212, 18446.5450804386, 5553.41745782321,  571.12530280711,   32.071168487829,
    54, 0.0228450603909931, 12.5291770375155, 4.02639127966407, 3.35611462366409, 10.9830924208576, 18401.9780501121, 5534.18125709636, 570.334467157792,  32.8982747184914,
    55, 0.0219820532251614, 12.7653472746447,  4.0672086769994, 3.35780080397842, 11.0073182585309, 18355.9843226533, 5505.67735676666, 569.408582187298,   33.690645385506,
    56, 0.0212643473687112, 12.9972996931488, 4.10984804355408, 3.35944705448582, 11.0360778189205, 18313.5666421319, 5476.57225915601, 568.456540664919,  34.4169386008352,
    57,  0.020522811420362, 13.2160977316657, 4.14971675773748, 3.36191692922136, 11.0673696407927, 18272.4923430153, 5448.08404811688, 567.602467352757,  35.1491165922329,
    58,  0.019764985939684, 13.4236807818853, 4.17821677654387, 3.36454231933092,  11.098517874254,   18220.79860584, 5418.78342955763, 566.442158774163,  35.9995839426757,
    59, 0.0191837759362101, 13.5716607938742, 4.18803740831452, 3.36635862432489, 11.1217783651542, 18156.8988392949, 5397.33913013441, 565.381691169765,  36.8936918351557,
    60, 0.0186690053379995, 13.6672680733148, 4.18602122448512, 3.36779830204151, 11.1376641921692, 18087.6320814357, 5382.56007826138, 564.655875327222,  37.8134327253513,
    61, 0.0181948378107616, 13.7623698067522, 4.17979713901931, 3.36977162838454, 11.1493680149158, 18012.0319894947,   5363.646746943, 564.564767204492,  38.8375179745024,
    62, 0.0179019814118502, 13.8465378013858, 4.18143554822188, 3.37161128119041, 11.1613241444812, 17924.1796221599, 5343.55193684101, 564.914532768304,  39.8505543268386,
    63, 0.0177484831511427, 13.9159018823301, 4.20440490849722, 3.37381207754384, 11.1840930855373, 17838.6079060573, 5327.02327016388, 565.717131279028,  40.7271365161339,
    64, 0.0176419724161205, 14.0231321031263, 4.26580003945328, 3.37780613013704, 11.2226261128336, 17764.0118806269, 5304.43032286467,  567.08195351907,  41.5505895791192,
    65,  0.017646463288716, 14.1668260430779, 4.35179075637024, 3.38306544311845, 11.2768363304236, 17691.0448553286, 5273.64344546029, 568.649122572307,  42.3248445003932,
    66, 0.0176701323211833, 14.3101337059625, 4.44918871638023, 3.38884962612537, 11.3343043572872, 17632.0910346348,  5247.1458254915, 569.664548751305,  42.9842847707338,
    67, 0.0176545419747317, 14.4630181639241, 4.53914877152224,  3.3961262718832, 11.3893200852875, 17600.1891128301, 5225.93923990759, 570.691167158565,  43.6406333156395,
    68, 0.0176791296523015, 14.6263230361894, 4.60544352025615, 3.40404069069328, 11.4207419622603, 17577.5252173155,  5206.4736845116, 571.936373965892,  44.3960631312051,
    69, 0.0177172588491117, 14.7598885180139, 4.62821205559482, 3.41076423806375, 11.4125928817152, 17555.0931429434, 5196.96403432574, 572.807290219713,  45.1830524901064,
    70, 0.0177239116489448,  14.871026655294,  4.6252013889399, 3.41686996058528, 11.3770041399834, 17537.7394350705, 5195.70584280841, 573.719163500735,  46.0169994438939,
    71, 0.0178586768682312, 15.0495153846765, 4.61700076792685, 3.42375850746965, 11.3306119068315, 17482.8527366382, 5169.26049386635, 575.718721295578,  47.0404281154674,
    72, 0.0180812232052713, 15.2749202569702, 4.62654501720617, 3.43043138170777, 11.2940511236839, 17385.6168437063, 5122.30354457397, 577.697621890416,  48.0249739941819,
    73, 0.0182718317998028, 15.5548090401055, 4.67985771808442,  3.4371511379872, 11.2975255200171, 17244.6441734209, 5049.10527782105, 579.433148281701,  48.8264546332972,
    74, 0.0185562739152061, 15.9470782355996, 4.80382801654026, 3.44494700928372, 11.3726213733875, 17039.1742139027, 4930.03557958124, 581.574760193174,  49.5221759208108,
    75, 0.0187645268469588,  16.351164638979,  4.9724389143518, 3.45114800181613, 11.5090086528761, 16808.8722627164, 4796.78880221689, 583.155937005295,  49.8965938550086,
    76,  0.018987881045584, 16.8168044784835, 5.18461544011601, 3.45735228472269, 11.7107448078775, 16541.1869937218, 4632.12456317306,  585.35130695975,  50.2169392423589,
    77, 0.0192808313626932,  17.305149186865, 5.39677945616887, 3.46417614446908, 11.9281526473082, 16263.5831500091,  4452.0359964465, 588.280895304657,  50.5792125659481,
    78, 0.0194630251771683, 17.7038389684913, 5.57311933813842, 3.46928582037981, 12.1337004432815, 16021.6273950353, 4291.51035148502, 591.141721869639,  50.9031653782253,
    79, 0.0195974981045352, 18.0455426973745, 5.72296058581763, 3.47278433081404, 12.3162508315001, 15796.6275552903, 4140.25291692508, 594.251910689782,  51.4398840988499,
    80, 0.0197968420553066, 18.3198081462551, 5.83260209425362, 3.47634173488182, 12.4527906532677, 15608.8441949289, 4011.11145734909, 597.291923572545,   52.313488419392,
    81,  0.019926045539777,  18.485472692746, 5.91824316826524, 3.47938141656407,   12.54103511433, 15479.9233025125, 3919.32207050158, 599.781797212348,  53.3142158515972,
    82, 0.0199190859140854, 18.5877177496203, 5.98784699055896, 3.48251002441543, 12.5936082192243, 15400.6269013959, 3862.67886826741, 601.522167913077,  54.3997054410783,
    83, 0.0199167124409372, 18.6801922648659, 6.05837571382971,  3.4880391403277, 12.6273907627874, 15361.3548128251, 3834.07546832001, 602.597418183909,  55.5857203176306,
    84,  0.019920713495381, 18.7622008226261, 6.11843840220738, 3.49525916997205, 12.6614040301772, 15344.9496967358, 3819.44891772192, 603.510653698308,  56.7097097574515,
    85, 0.0196944597855939, 18.8415158755615, 6.15008649209526, 3.50310616161617, 12.6888219512048, 15353.2857452574, 3819.45489879938, 604.416791346305,  57.7717981507999,
    86, 0.0194364543323017, 18.9391542666598, 6.15293230582037, 3.51150641930423, 12.7223956406912, 15374.9507388447, 3830.28668915245, 605.304948331809,  58.8022018143519,
    87, 0.0189826589548353,  19.038489755394, 6.12253711594421,  3.5175231599918, 12.7556842817255, 15395.0313124511, 3822.44829112387, 606.231839065589,  59.7397559755178,
    88, 0.0181182846834245, 19.1223777635494, 6.05720155568019, 3.52011073488561, 12.7882650328948,  15421.290802669, 3795.72137438125, 607.115348163618,  60.6423377166732,
    89, 0.0174216921782884, 19.2092119542286,  6.0123797664954, 3.52288163469841, 12.8545051889396, 15450.6573043823,  3773.2658798852, 608.001637077996,  61.5710969860888,
    90, 0.0167744921865688, 19.3008973502212, 6.04681428418366, 3.52533394480166, 12.9802389716051, 15477.6641628946, 3745.52115313336, 609.310531313389,  62.5641803645149,
    91, 0.0160793627061195, 19.3901358092061, 6.13590049226483, 3.52725067293369, 13.1472281116035, 15504.0394850499, 3713.15837764339, 610.683634096237,  63.6177990490524,
    92, 0.0153329292383867, 19.4685689970553, 6.23990102593015, 3.52905289376498, 13.2632422070239, 15530.8210068474,  3682.9140150122, 611.881932539036,  64.7405963564943,
    93,    0.0145497203871, 19.5445768155406, 6.36635191608606, 3.53056964968893, 13.3776508911495, 15557.1562153283, 3650.48853467762, 613.104573725049,  65.9214660068371,
    94, 0.0137121449471466, 19.6210929394864, 6.48820821332382, 3.53167984316096, 13.5134829069908, 15582.5289060255,   3612.111191651, 614.257902481167,  67.1718902318194,
    95, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    96, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    97, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    98, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    99, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    100, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    101, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    102, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    103, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    104, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    105, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    106, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    107, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    108, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    109, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    110, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    111, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    112, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    113, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    114, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    115, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    116, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    117, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    118, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    119, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368,
    120, 0.0127392345098101, 19.6929881557982, 6.63228835360714, 3.53203355069165,  13.655114902985, 15606.5083059275, 3566.69417819592, 615.453286615824,  68.4940364133368
  )
  
  return(driver_averages)
  
}
  driver_averages <- driver_averages_fct()

add_driver_averages <- function(input){

  input <- left_join(input,driver_averages, by = "drv_age1")
  
  input <- input %>% mutate(AS_drv_ncd         = pol_no_claims_discount - avg_ncd)
  input <- input %>% mutate(AS_drv_pol         = pol_duration - avg_pol)
  input <- input %>% mutate(AS_drv_polsit      = pol_sit_duration - avg_polsit)
  input <- input %>% mutate(AS_drv_pco         = pc - avg_pco)
  input <- input %>% mutate(AS_drv_nyl         = drv_nyl1 - avg_nyl)  
  input <- input %>% mutate(AS_drv_vh_age      = vh_age - avg_vh_age)
  input <- input %>% mutate(AS_drv_vh_value    = vh_value - avg_vh_value) 
  input <- input %>% mutate(AS_drv_vh_valuedep = vh_valuedep - avg_vh_valuedep)
  input <- input %>% mutate(AS_drv_pop         = population - avg_pop)  

  
  input <- input %>% mutate(AD_drv_ncd         = pol_no_claims_discount / avg_ncd)
  input <- input %>% mutate(AD_drv_pol         = pol_duration / avg_pol)
  input <- input %>% mutate(AD_drv_polsit      = pol_sit_duration / avg_polsit)
  input <- input %>% mutate(AD_drv_pco         = pc / avg_pco)
  input <- input %>% mutate(AD_drv_nyl         = drv_nyl1 / avg_nyl)
  input <- input %>% mutate(AD_drv_vh_age      = vh_age / avg_vh_age)
  input <- input %>% mutate(AD_drv_vh_value    = vh_value / avg_vh_value) 
  input <- input %>% mutate(AD_drv_vh_valuedep = vh_valuedep / avg_vh_valuedep)   
  input <- input %>% mutate(AD_drv_pop         = population / avg_pop) 
  
  return(input)
}


nyl_averages_fct <- function(){
  
  # I have computed various averages for each Driver Ages.
  # We will use those as a metrics on how far from the "average" a specific client is.
  # I did fit a smooth curve (loess) to get the best proxy of the general population
  # I used package datapasta to convert into code.
  
  nyl_averages <- tibble::tribble(
    ~drv_nyl1,           ~Navg_ncd,         ~Navg_pol,      ~Navg_polsit,         ~Navg_pco,      ~Navg_vh_age,    ~Navg_vh_value, ~Navg_vh_valuedep,         ~Navg_age,
    1,  0.409766712516377, 4.18565899461085, 2.45681497169379, 2.98520690369971, 10.7631410485527, 13721.5553013566, 4092.32656670796, 32.7598418716406,
    2,  0.398090649204697, 4.40262903611012, 2.61509262836928, 3.00355912680048, 10.8327082216479, 14053.5709114274, 4156.51220778278, 31.6975043456361,
    3,  0.384988297904014, 4.62275025716975, 2.75335069416593, 3.02519225419389, 10.8850232020568, 14379.1880590758, 4229.45109819873,  30.904272429038,
    4,  0.371795763517469, 4.84766149901332, 2.91161746123478, 3.04964284744192, 10.9581654158098, 14698.6098392104, 4311.33258827142, 30.3849231129595,
    5,  0.355825861980138, 5.07355050609645, 3.03437885238046, 3.07636608011594,  11.002747843255, 15008.4606296639, 4398.28298910867, 30.1188302842859,
    6,  0.334424951136076, 5.29450253212041, 3.08541180105874, 3.10654345105575, 10.9870183992314, 15310.2763682311, 4493.25507446348, 30.1154649895436,
    7,  0.308418442803745, 5.51866622930712, 3.13219812195477, 3.13881111205768, 10.9693467027386, 15601.8851148907, 4593.07446763293, 30.3561402290759,
    8,  0.281062792425917, 5.74091418128287, 3.15615717719175, 3.17399558743752, 10.9131170395934, 15883.7474820342, 4696.43593796498, 30.9494480628306,
    9,  0.251297548556046,  5.9854250972086, 3.21696913204726, 3.20734536447419, 10.8842495399395, 16162.5891550991, 4810.93978051828, 31.7127746124617,
    10,  0.223996254011435,  6.2488022520496, 3.28286170798239, 3.23886714504579, 10.8230236888449,  16430.704731473, 4928.36357321664, 32.6120228415028,
    11,  0.198847783243671, 6.53928990347858, 3.34973634388039, 3.26765222730877, 10.7335660483436, 16675.0711105749, 5041.97023917001, 33.4831981577257,
    12,  0.173629272066442, 6.83658100727144, 3.41752647770323, 3.29268784279651, 10.6438333479075, 16898.5432588143, 5154.47955211331, 34.3874456713867,
    13,   0.15023011303545, 7.12219237400157, 3.47378539830294, 3.31049130054548,  10.557883200672, 17103.7522569943, 5255.03748162969, 35.2885771611754,
    14,  0.127823742925142, 7.40847205816471, 3.52407976789002, 3.32052692112997, 10.4997089484812, 17272.2292417997, 5331.71578549873, 36.1900814300081,
    15,  0.109136015469185, 7.69265067882916, 3.55611510616159, 3.32684651020569, 10.4722311473087, 17424.7774738818, 5398.73148808363, 37.0971366730322,
    16, 0.0932870476040041, 7.99911563920903, 3.57784649764247, 3.33211900567191, 10.4683175431439, 17585.1149035359, 5462.94582141322, 37.9999065726769,
    17, 0.0809513937457851, 8.31580165502155, 3.60033310009152, 3.33584226912627, 10.4884054163207, 17735.8119596121, 5515.28067471352, 38.9073442481007,
    18, 0.0715812487401122, 8.63513774965435, 3.62672662608233, 3.34226683034977, 10.5284842814216, 17876.6866266954, 5560.19090323888, 39.8136418735063,
    19, 0.0631691479402795, 8.96943846629287,  3.6701427730667, 3.34784837695554, 10.5754060609608, 18007.5568893707, 5602.13136224366, 40.7289876672655,
    20,  0.056030011974343, 9.29039626485393, 3.72266328899221, 3.35178411069686, 10.6237760443187, 18144.4068191957, 5652.19079714889, 41.6566445575321,
    21, 0.0496599880208778, 9.60704139171282, 3.77827713332599, 3.35465514474856, 10.6647563545018, 18281.7985944521, 5704.40646172589, 42.6028498019906,
    22, 0.0442088600274606, 9.91933556046454, 3.83335117653126, 3.35401780668127, 10.7001521943619,  18387.417359811,  5737.6559737544, 43.5816647743362,
    23, 0.0396165680507677, 10.2382597323843, 3.88150617768745, 3.35144777039411, 10.7367171091473, 18459.8625158273, 5751.92742435797,  44.598743957035,
    24, 0.0359339176243164, 10.5406501542401, 3.91628368085456, 3.34851659578265, 10.7779371560012, 18511.2578874212, 5754.27315233588, 45.6365734073378,
    25, 0.0330261080213287, 10.8355563468498, 3.94013526199557, 3.34815720501127, 10.8244494997106, 18521.6914545715, 5726.32465950178,   46.70096552634,
    26, 0.0303512245114078, 11.1305226061442, 3.95762490910134,  3.3489116293964, 10.8858100787292, 18493.4854786962, 5670.84929478888, 47.7741972844533,
    27, 0.0280253272938274, 11.3923712255939, 3.97484280562522, 3.35352142261145, 10.9510092603618, 18447.9631631745, 5609.02481816983, 48.8423391776195,
    28,  0.026078003074818, 11.6483239535688, 3.99462943405217, 3.35890888998526, 11.0122699583154,  18406.447711385, 5562.02898961725, 49.9046819706259,
    29, 0.0244325718088183, 11.8998895076079, 4.01042151555922, 3.36308061748646,  11.062548683579, 18361.2425700769, 5530.90901654869,  50.937673383041,
    30,  0.023237311643824, 12.1306431609321, 4.02540660308106, 3.36691777646591, 11.0884834559294, 18302.3809084345, 5501.78492358781, 51.9579325843405,
    31, 0.0224205440986288, 12.3678894845094, 4.04212633014264, 3.36882995620091, 11.0940699462737, 18247.7805134897,  5473.4436965164, 52.9553055595298,
    32, 0.0217737364158388,  12.592320829369, 4.06065666458221, 3.37071061066374, 11.0817286697544, 18208.8694449562,  5461.6391498182, 53.9389312934229,
    33, 0.0211779082364607, 12.8126811127243, 4.08386967430857, 3.37260239797926, 11.0612319012959, 18168.5196206654, 5449.82099679212, 54.9233171623318,
    34, 0.0206102835494522, 13.0100682110898, 4.11340550343746,  3.3760097463516,  11.048819826691, 18115.6099447949,  5417.7842521494, 55.9188717504605,
    35,   0.02000467871888,  13.216375982195, 4.15779497764021, 3.37914164805663, 11.0587071800102, 18057.8719801418, 5380.91964622032, 56.9299727542258,
    36, 0.0193850909427016, 13.4193697335895, 4.20676704620119, 3.38112896100506, 11.0885275832698, 17998.6682888435, 5342.95158643799, 57.9884396178992,
    37, 0.0187427105801494,  13.615702403203, 4.25475212945013, 3.38288518481344, 11.1449201505188, 17941.3614330372, 5307.60448023553, 59.0647381532684,
    38,  0.018074358739726, 13.8075068528095, 4.30087024951565, 3.38297953047845,  11.209001172799, 17881.7784616464, 5263.49585116025, 60.1806673117527,
    39, 0.0174251856432523, 13.9749529242339, 4.33685427348923, 3.38346907775745, 11.2686271976614, 17818.9765791539,  5211.5036742146, 61.2956606917403,
    40, 0.0168317402443174, 14.1272875838636, 4.36685774978659, 3.38516603766102, 11.3186945351504, 17761.1635810362,  5173.7430340144, 62.3928352705682,
    41, 0.0162964737705438, 14.2693787822604, 4.38897486991171, 3.38941521545695, 11.3507528497589, 17711.0552355665, 5161.06228248286, 63.4261213356628,
    42,  0.015846880529548, 14.4085334439552, 4.41145093897902, 3.39438711661243, 11.3675353743829, 17662.7388804685, 5153.83646824121, 64.4392323106633,
    43, 0.0155162381915184, 14.5613539183531, 4.44720874326755, 3.40053307422147, 11.3832036226899, 17619.7099201338, 5152.93159004445, 65.3969531395382,
    44, 0.0153833128369758, 14.7380333606343, 4.49759011328671, 3.40709017188744,  11.407013165839, 17583.2123397123,  5164.9934652369, 66.3587511856457,
    45, 0.0155044112303732, 14.9339580908634, 4.56156381276606, 3.41169689721828, 11.4325074791323,  17545.134159993, 5175.09695136545, 67.3718545117896,
    46, 0.0158572554683611, 15.1601879202238, 4.64673895894559,  3.4161394843527, 11.4662221492314, 17497.3634017648, 5168.31690597701, 68.4133094355993,
    47,  0.016146262113595, 15.3849351386952, 4.74016351534645, 3.42076976180789, 11.5102941811581, 17433.6410416755, 5138.86041212593, 69.5525943231397,
    48, 0.0163598635578396, 15.6160191839091, 4.83387241176814, 3.42568482604959, 11.5550153632878, 17362.1458375249, 5099.59242926482, 70.7034117260471,
    49, 0.0164135423902562,  15.832708388837, 4.91554152042995, 3.43270881218371, 11.5891438178284, 17296.4224818295,   5063.574629538,  71.875992023231,
    50, 0.0164318262461382, 16.0480734874024, 4.98051541388633, 3.43967175186859, 11.6048396339066, 17231.5621085984, 5030.70386885526, 72.9789456998492,
    51, 0.0163763694534539, 16.2359152083192, 5.02593994411616, 3.44601635445233, 11.6006374520086, 17169.8128702027,  4994.7988311622,  74.027183576009,
    52, 0.0163349686385584, 16.4326170027571, 5.06915539895024, 3.44985248158256, 11.5980062308439, 17122.9909832229, 4961.28491011452, 74.9268453317789,
    53, 0.0165833776075062, 16.6267510783906, 5.11431273229893, 3.45298288256961, 11.5882590129043, 17081.7004038306, 4928.08519334515, 75.7792888672773,
    54, 0.0168864014506486, 16.8259464524084,  5.1701969730073, 3.45573189933092, 11.5960613425827, 17038.4029614711, 4887.89063601809, 76.5687618472896,
    55, 0.0171400653552066, 17.0552242599982,  5.2474876972895,  3.4580900610508, 11.6429364636266,   16985.56048559, 4833.39219329736, 77.3613069033372,
    56, 0.0173130676171107, 17.2748515233007, 5.34219649531555, 3.45987513257287, 11.7191590191824, 16918.9373872921, 4761.80450401396, 78.2357861429029,
    57, 0.0172592962280731, 17.5068031565397,  5.4527162945288, 3.46124279083639, 11.8308427816125, 16850.3894436513,   4683.025470908, 79.1359184286432,
    58, 0.0173309040860885, 17.7060729393638, 5.56682006248012, 3.46326241503195, 11.9393923512599, 16796.5155330659, 4608.77094500716, 80.1205228684681,
    59, 0.0175665611587732, 17.8866071401279,  5.6693401879617, 3.46529915340925, 12.0188190943819, 16727.4457547013, 4507.07064577216, 81.1249987887414,
    60, 0.0181527343818371, 18.0190854402749, 5.75624665536691, 3.46626491984869, 12.0574024697114, 16684.6195839216, 4417.31012884029, 82.1010309338948,
    61, 0.0185998339389076, 18.1474590058249, 5.82321611005842, 3.46332935816444,  12.079638858327, 16701.8430137909, 4378.32889286309, 82.9658925173751,
    62, 0.0189358248116362, 18.2449020085285, 5.86613627014598, 3.45964582216749, 12.0792064711986, 16732.2466967684, 4350.53324260892, 83.7875925533863,
    63, 0.0187815883611841, 18.3409038925829, 5.90530262794282,  3.4556253119884, 12.0967606257826, 16778.3805646438, 4335.02299254556, 84.5087046063109,
    64, 0.0178970601496831, 18.4553450995845, 5.94006821577318, 3.45258618285934, 12.1454691503859, 16842.7945492067, 4332.89795714076, 85.2189507665003,
    65, 0.0173256163802255, 18.5667331481793, 5.97779470253475, 3.45322923072877, 12.2175873475793,  16913.364894223, 4334.84410174597, 85.9715346066656,
    66, 0.0169741439753071, 18.6992233256801, 6.02857097242683, 3.45307781009064, 12.3210489479444, 16978.1489233208, 4331.67696525506, 86.7395295464619,
    67, 0.0177303772051417,    18.8241638758, 6.06767104881823, 3.45188375849382, 12.4409535320188, 17039.9681880834, 4324.69072244951, 87.5493674136724,
    68, 0.0185734853822701, 18.9479287902228,  6.0846838855227, 3.44527209993607,  12.552182821232, 17100.3279438866, 4312.58279840367, 88.4194593841494,
    69, 0.0198940094270797, 19.0638710840004, 6.07018293176482, 3.43622139025895, 12.6570535577185, 17157.5598807439,  4297.2079196066,  89.378114347175,
    70, 0.0189482024783455, 19.1808556340638, 6.04411610829665, 3.42510524137213, 12.7244855448655, 17198.0208948645, 4273.59514008366, 90.4348211473563,
    71, 0.0169861824338908, 19.2943850234538, 6.01234672446366, 3.41141201511018,   12.77873799494, 17216.6380253735, 4236.85743034401, 91.5747810938984,
    72, 0.0159312855238304, 19.4008521598086, 5.98147561793439, 3.39402353355816, 12.8492907846536, 17224.3276901848, 4193.15079201689, 92.8047311696045,
    73,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    74,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    75,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    76,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    77,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    78,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    79,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    80,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    81,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    82,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    83,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    84,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    85,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    86,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    87,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    88,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    89,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    90,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    91,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    92,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    93,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    94,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    95,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    96,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    97,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    98,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    99,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533,
    100,  0.014390765742539, 19.4990626266829, 5.94731357498662, 3.37362768519724, 12.9148454854425, 17232.0063072125, 4148.63122673152, 94.1262728744533
  )
  
    
    
  return(nyl_averages)
}
  nyl_averages <- nyl_averages_fct()


      
add_nyl_averages <- function(input){
    
  
    input <- left_join(input,nyl_averages, by = "drv_nyl1")

    input <- input %>% mutate(AS_nyl_ncd         = pol_no_claims_discount - Navg_ncd)
    input <- input %>% mutate(AS_nyl_pol         = pol_duration - Navg_pol)
    input <- input %>% mutate(AS_nyl_polsit      = pol_sit_duration - Navg_polsit)
    input <- input %>% mutate(AS_nyl_pco         = pc - Navg_pco)
    input <- input %>% mutate(AS_nyl_age         = drv_age1 - Navg_age)  
    input <- input %>% mutate(AS_nyl_vh_age      = vh_age - Navg_vh_age)
    input <- input %>% mutate(AS_nyl_vh_value    = vh_value - Navg_vh_value)  
    input <- input %>% mutate(AS_nyl_vh_valuedep = vh_valuedep - Navg_vh_valuedep)   
  
    input <- input %>% mutate(AD_nyl_ncd         = pol_no_claims_discount / Navg_ncd)
    input <- input %>% mutate(AD_nyl_pol         = pol_duration / Navg_pol)
    input <- input %>% mutate(AD_nyl_polsit      = pol_sit_duration / Navg_polsit)
    input <- input %>% mutate(AD_nyl_pco         = pc / Navg_pco)
    input <- input %>% mutate(AD_nyl_age         = drv_age1 / Navg_age)
    input <- input %>% mutate(AD_nyl_vh_age      = vh_age / Navg_vh_age)
    input <- input %>% mutate(AD_nyl_vh_value    = vh_value / Navg_vh_value)  
    input <- input %>% mutate(AD_nyl_vh_valuedep = vh_valuedep / Navg_vh_valuedep) 
    
    return(input)
}  

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8888888888                888                                      
# 888                       888                                      
# 888                       888                                      
# 8888888  .d88b.   8888b.  888888 888  888 888d888 .d88b.  .d8888b  
# 888     d8P  Y8b     "88b 888    888  888 888P"  d8P  Y8b 88K      
# 888     88888888 .d888888 888    888  888 888    88888888 "Y8888b. 
# 888     Y8b.     888  888 Y88b.  Y88b 888 888    Y8b.          X88 
# 888      "Y8888  "Y888888  "Y888  "Y88888 888     "Y8888   88888P'  
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#### FEATURES ####

# for the final test, we will need to have a data saved with all the information, in order to score year 5

generating_features_bf_encoding <- function(df){
  
  cat("...Generating Features using Lag :: start ")
  # identifying pay_freq change
  f_pay <- df %>% 
    group_by(id_policy) %>% 
    mutate(prev.pol_pay_freq = dplyr::lag(pol_pay_freq, order_by=id_policy))    
  
  f_pay  <- f_pay %>% mutate(prev.pol_pay_freq = ifelse(is.na(prev.pol_pay_freq), pol_pay_freq, prev.pol_pay_freq))
  f_pay  <- f_pay %>% mutate(F_pol_pay_freq_change = ifelse(pol_pay_freq != prev.pol_pay_freq, 1, 0))
  
  f_pay_final  <- f_pay %>%  ungroup() %>%  select(F_pol_pay_freq_change)
  
  
  # identifying population change
  f_move <- df %>% 
    group_by(id_policy) %>% 
    mutate(prev.population = dplyr::lag(population, order_by=id_policy))    
  
  f_move  <- f_move %>% mutate(prev.population = ifelse(is.na(prev.population), population, prev.population))
  f_move  <- f_move %>% mutate(F_client_has_moved1 = ifelse(population != prev.population, 1, 0))
  
  # identifying town_surface_area change
  f_move2 <- df %>% 
    group_by(id_policy) %>% 
    mutate(prev.town = dplyr::lag(town_surface_area, order_by=id_policy))    
  
  f_move2  <- f_move2 %>% mutate(prev.town = ifelse(is.na(prev.town), town_surface_area, prev.town))
  f_move2  <- f_move2 %>% mutate(F_client_has_moved2 = ifelse(town_surface_area != prev.town, 1, 0))
  
  f_move$F_client_has_moved2 <- f_move2$F_client_has_moved2
  f_move <- f_move %>%  mutate(F_client_has_moved = ifelse((F_client_has_moved1 == 1 | F_client_has_moved2 == 1), 1, 0))
  
  f_move_final  <- f_move %>%  ungroup() %>%  select(F_client_has_moved)
  
  
  # No Claims Discount ....going down the rabbit hole
  
  f_ncd <- df %>%
    group_by(id_policy) %>%
    mutate(prev.pol_no_claims_discount = dplyr::lag(pol_no_claims_discount, order_by=id_policy))
  
  f_ncd  <- f_ncd %>% mutate(pol_ncd_occ = ifelse(pol_no_claims_discount > prev.pol_no_claims_discount, 1, 0))
  f_ncd  <- f_ncd %>% mutate(pol_ncd_noocc = ifelse(pol_no_claims_discount <= prev.pol_no_claims_discount, 1, 0))
  f_ncd  <- f_ncd %>% mutate(pol_ncd_inc = ifelse(pol_no_claims_discount > prev.pol_no_claims_discount, pol_no_claims_discount-prev.pol_no_claims_discount, 0))
  f_ncd  <- f_ncd %>% mutate(pol_ncd_dec = ifelse(pol_no_claims_discount < prev.pol_no_claims_discount, pol_no_claims_discount-prev.pol_no_claims_discount, 0))
  f_ncd  <- f_ncd %>% mutate(pol_ncd_expo = pol_ncd_occ + pol_ncd_noocc )
  
  
  f_ncd  <- f_ncd %>% mutate(pol_ncd_inc_risk = case_when(pol_ncd_inc>0.35 ~ 2.0,
                                                          (pol_ncd_inc>0.25 & pol_ncd_inc <=0.35)~1.5,
                                                          (pol_ncd_inc>0.15 & pol_ncd_inc <=0.25)~1.0,
                                                          (pol_ncd_inc>0 & pol_ncd_inc <=0.15)~0.5,
                                                          pol_ncd_inc==0 ~ 0))
  
  f_ncd  <- f_ncd %>% mutate(pol_ncd_dec_risk = case_when(pol_ncd_dec < -0.10 ~ 2,
                                                          (pol_ncd_dec < -0.03 & pol_ncd_dec >= -0.10) ~1,
                                                          (pol_ncd_dec < 0 & pol_ncd_dec >= -0.03) ~ 0.5,
                                                          (pol_ncd_dec == 0) ~0))
  
  f_ncd_c <- f_ncd %>%
    group_by(id_policy) %>%
    mutate(pol_ncd_expo_c = cumsum(replace_na(pol_ncd_expo,0)),
           pol_ncd_occ_c = cumsum(replace_na(pol_ncd_occ,0)),
           pol_ncd_noocc_c = cumsum(replace_na(pol_ncd_noocc,0)),
           pol_ncd_inc_risk_c = cumsum(replace_na(pol_ncd_inc_risk,0)), 
           pol_ncd_dec_risk_c = cumsum(replace_na(pol_ncd_dec_risk,0)))  
  
  f_ncd_c <- f_ncd_c %>% 
    mutate(pol_ncd_inc_risk_ca = ifelse(pol_ncd_expo_c == 0, NA, pol_ncd_inc_risk_c / pol_ncd_expo_c),
           pol_ncd_dec_risk_ca = ifelse(pol_ncd_expo_c == 0, NA, pol_ncd_dec_risk_c / pol_ncd_expo_c))
  
  f_ncd_c_final <- f_ncd_c %>% ungroup() %>% 
    select(pol_ncd_occ, pol_ncd_noocc, pol_ncd_occ_c, pol_ncd_noocc_c,
           pol_ncd_inc, pol_ncd_dec, 
           pol_ncd_inc_risk, pol_ncd_dec_risk,
           pol_ncd_inc_risk_c, pol_ncd_dec_risk_c,
           pol_ncd_inc_risk_ca, pol_ncd_dec_risk_ca)
  
  
  
  ffs <- cbind(f_pay_final,f_move_final,f_ncd_c_final)
  
  df_features_bf_encoding <- data.frame(ffs)
  
  #because on year 1, some variable will be strictly NAs and be defined as a "logical" variable,
  #I need to force the variables to be numberic.
  cols <- sapply(df_features_bf_encoding, is.logical)
  df_features_bf_encoding[,cols] <- lapply(df_features_bf_encoding[,cols], as.numeric)
  
  #I will take care of the NAs here, impuing the median
  df_features_bf_encoding  <- df_features_bf_encoding %>% 
    mutate(pol_ncd_occ           = ifelse(is.na(pol_ncd_occ),          0,pol_ncd_occ),
           pol_ncd_noocc         = ifelse(is.na(pol_ncd_noocc),        1,pol_ncd_noocc),
           pol_ncd_occ_c         = ifelse(is.na(pol_ncd_occ_c),        0,pol_ncd_occ_c),
           pol_ncd_noocc_c       = ifelse(is.na(pol_ncd_noocc_c),      1,pol_ncd_noocc_c),
           pol_ncd_inc           = ifelse(is.na(pol_ncd_inc),          0,pol_ncd_inc),
           pol_ncd_dec           = ifelse(is.na(pol_ncd_dec),          0,pol_ncd_dec),
           pol_ncd_inc_risk      = ifelse(is.na(pol_ncd_inc_risk),     0,pol_ncd_inc_risk),
           pol_ncd_dec_risk      = ifelse(is.na(pol_ncd_dec_risk),     0,pol_ncd_dec_risk),
           pol_ncd_inc_risk_c    = ifelse(is.na(pol_ncd_inc_risk_c),   0,pol_ncd_inc_risk_c),
           pol_ncd_dec_risk_c    = ifelse(is.na(pol_ncd_dec_risk_c),   0,pol_ncd_dec_risk_c),
           pol_ncd_inc_risk_ca   = ifelse(is.na(pol_ncd_inc_risk_ca),  0,pol_ncd_inc_risk_ca),
           pol_ncd_dec_risk_ca   = ifelse(is.na(pol_ncd_dec_risk_ca),  0,pol_ncd_dec_risk_ca),
           F_pol_pay_freq_change = ifelse(is.na(F_pol_pay_freq_change),0,F_pol_pay_freq_change),
           F_client_has_moved    = ifelse(is.na(F_client_has_moved),   0,F_client_has_moved)      )  
  
  #str(df_features_bf_encoding)
  
  #df_features_bf_encoding
  
  cat("......... done ")
  
  return(df_features_bf_encoding)
}

generating_features <- function(df){
  
  # Historical claim count indicator
  # indicator for prev year, last 2 years, all
  # claim over value  (an indicator of link between $ claim and veh value)
  
  f_claims <- df %>%
    group_by(id_policy) %>%
    mutate(claim_cnt = claim_nb,
           claim_cnt_c = cumsum(claim_nb), 
           claim_amt_c = cumsum(claim_amount),
           vh_value_avg = cummean(vh_value))
  
  f_claims <- f_claims %>% 
    group_by(id_policy, year) %>% 
    summarise(F_claim_cnt = claim_cnt,
              F_claim_cnt_c = sum(claim_cnt_c), 
              F_claim_amt_c = sum(claim_amt_c),
              vh_value_avg = mean(vh_value_avg))   
  
  
  f_claims
  f_claims$F_claim_amt_avg = f_claims$F_claim_amt_c / f_claims$F_claim_cnt_c
  f_claims$F_cov = f_claims$F_claim_amt_avg / f_claims$vh_value_avg  
  
  # replace na with zero
  f_claims  <- f_claims %>% mutate(F_claim_amt_avg = ifelse(is.na(F_claim_amt_avg), 0, F_claim_amt_avg),
                                   F_cov = ifelse(is.na(F_cov), 0, F_cov))
  
  f_claims <- f_claims %>% select(-vh_value_avg)  
  
  # change year to + 1, to push the data to the next year, as feeding info
  f_claims$year = f_claims$year + 1
  

  df_features <- left_join(f_claims, f_move,  by = c("id_policy","year") )
  
  return(df_features)
}

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 8888888b.                  8888888b.                                                     
# 888   Y88b                 888   Y88b                                                    
# 888    888                 888    888                                                    
# 888   d88P 888d888 .d88b.  888   d88P 888d888 .d88b.   .d8888b .d88b.  .d8888b  .d8888b  
# 8888888P"  888P"  d8P  Y8b 8888888P"  888P"  d88""88b d88P"   d8P  Y8b 88K      88K      
# 888        888    88888888 888        888    888  888 888     88888888 "Y8888b. "Y8888b. 
# 888        888    Y8b.     888        888    Y88..88P Y88b.   Y8b.          X88      X88 
# 888        888     "Y8888  888        888     "Y88P"   "Y8888P "Y8888   88888P'  88888P'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#### PREPROCESS ####

prepreprocess_X_data <- function (x_raw, leakage_blocker, keep_vh_make_model){
  
  cat("..Prepreprocess X data :: start \n")
  
  td <- x_raw #train_data
  
  td <- td %>% mutate(obs = 1)
  
  td <- td %>% mutate(policy_cnt = 1)
  
  x_id_policy <- td$id_policy
  x_vh_make_model <- td$vh_make_model
  
  features_x_bf_encoding <- generating_features_bf_encoding(td)  
  
  #cat("Names of features data field just after being processed:\n")
  #print(names(features_x_bf_encoding))
  # features_x_bf_encoding <- features_x_bf_encoding %>% select(pol_ncd_occ, pol_ncd_noocc, pol_ncd_occ_c, pol_ncd_noocc_c,
  #                                                             pol_ncd_inc, pol_ncd_dec, 
  #                                                             pol_ncd_inc_risk, pol_ncd_dec_risk,
  #                                                             pol_ncd_inc_risk_c, pol_ncd_dec_risk_c,
  #                                                             pol_ncd_inc_risk_ca, pol_ncd_dec_risk_ca,
  #                                                             F_pol_pay_freq_change)
  
  td <- cbind(td,features_x_bf_encoding)
  
  #cat("Names of features data field after merging with all the data:\n")
  #print(names(td))
  
  
  #td <- td %>% mutate(pol_coverage.Max = ifelse(pol_coverage="Max", 1, 0)
  #td <- td %>% mutate(pol_coverage.Med1 = ifelse(pol_coverage="Med))
  
  
  
  #In order to create an average metric whenever there's a secondary driver. I'm using 0.7 and 0.3 weights.
  td <- td %>% 
    mutate(wgt_drv_age1 = ifelse(drv_drv2 == "No", 1, 0.7 ))
  td <- td %>% 
    mutate(wgt_drv_age2 = ifelse(drv_drv2 == "No", 0, 0.3 ))
  
  
  td <- td %>% mutate(drv_age2 = replace_na(drv_age2,0))
  td <- td %>% mutate(drv_age_lic2 = replace_na(drv_age_lic2,0))


  
  if (leakage_blocker == TRUE) { 
    
        td <- td %>% mutate(pc  = case_when(pol_coverage == "Max"  ~ 4,
                                            pol_coverage == "Med1" ~ 2,
                                            pol_coverage == "Med2" ~ 3,
                                            pol_coverage == "Min"  ~ 1,
                                           TRUE ~ 2))

        
        td <- td %>% mutate(pu = case_when(pol_usage == "AllTrips"     ~ 2,
                                           pol_usage == "Professional" ~ 1.50,
                                           pol_usage == "Retired"      ~ 1.05,
                                           pol_usage == "WorkPrivate"  ~ 1,
                                           TRUE ~ 1))
        
        td <- td %>% mutate(vf = case_when(vh_fuel == "Diesel"   ~ 2.0,
                                           vh_fuel == "Gasoline" ~ 1.0,
                                           vh_fuel == "Hybrid"   ~ 2.0,
                                           TRUE ~ 1.4))
  }
  
  else if (leakage_blocker == FALSE){
        td <- td %>% mutate(pc  = case_when(pol_coverage == "Max"  ~ 4,
                                            pol_coverage == "Med1" ~ 2,
                                            pol_coverage == "Med2" ~ 3,
                                            pol_coverage == "Min"  ~ 1,
                                            TRUE ~ 2))
        
        td <- td %>% mutate(pu = case_when(pol_usage == "AllTrips"     ~ 2.00,
                                           pol_usage == "Professional" ~ 1.50,
                                           pol_usage == "Retired"      ~ 1.05,
                                           pol_usage == "WorkPrivate"  ~ 1,
                                           TRUE ~ 1))
        
        td <- td %>% mutate(vf = case_when(vh_fuel == "Diesel"   ~ 2.0,
                                           vh_fuel == "Gasoline" ~ 1.0,
                                           vh_fuel == "Hybrid"   ~ 2.0,
                                           TRUE ~ 1.4))   
  }
  
  
  
  td$pcpu  = td$pc * td$pu
  td$pcppu = td$pc + td$pu
  

  #Converting some string var into factors
  td$pol_coverage <- as.factor(td$pol_coverage)
  td$pol_pay_freq <- as.factor(td$pol_pay_freq)
  td$pol_payd <- as.factor(td$pol_payd)
  td$pol_usage <- as.factor(td$pol_usage)
  td$drv_sex1 <- as.factor(td$drv_sex1)
  td$drv_drv2 <- as.factor(td$drv_drv2)
  td$drv_sex2 <- as.factor(td$drv_sex2)
  td$vh_fuel <- as.factor(td$vh_fuel)
  td$vh_type <- as.factor(td$vh_type)
  
  
  # Inflating claim_amount to current $
  # 1.010846, 1.021810, 1.032892, 1.044095
  
  from_1_to_4 <- (1.044095 / 1.010846)
  from_2_to_4 <- (1.044095 / 1.021810)
  from_3_to_4 <- (1.044095 / 1.032892)
  from_4_to_4 <- 1
  from_5_to_4 <- 0.975
  
  year <- c(1,2,3,4,5)
  infl_factor <- c(from_1_to_4,from_2_to_4,from_3_to_4, from_4_to_4, from_5_to_4)
  
  infl_mapping <- data.frame(year,infl_factor)
  infl_mapping
  
  td <- left_join(td, infl_mapping, by = "year")
  
  #  there are features x1, x2, x3, x4, x5, , xn, create features x1-x2, x1-x3, , x2-x3, x2-x4,  and x1/x2, x1/x3,
  
  
  # Adding an indicator of policy info charact and how info is up to date
  td$pol_sit_o_duration = td$pol_sit_duration / td$pol_duration
  td$pol_duration_m_sit = td$pol_duration - td$pol_sit_duration   
  
  
  # NYL
  td$drv_nyl1 = td$drv_age_lic1
  td$drv_nyl2 = td$drv_age_lic2
  
  td$drv_lic1 = td$drv_age1 - td$drv_age_lic1
  td$drv_lic2 = td$drv_age2 - td$drv_age_lic2
  
 
  #to account for NAs in the drv_age2, I will tweak it to be equal to drv_age1 whenever it's NA.
  #my logic is that it can be seen as 2 drivers with same age driving all the time.
  
  # Putting back the ages of empty 2nd driver to NA
  td <- td %>% mutate(drv_age2     = ifelse(drv_age2     == 0, drv_age1, drv_age2))
  td <- td %>% mutate(drv_age_lic2 = ifelse(drv_age_lic2 == 0, drv_age_lic1, drv_age_lic2))
  td <- td %>% mutate(drv_nyl2     = ifelse(drv_nyl2     == 0, drv_nyl1, drv_nyl2))
  td <- td %>% mutate(drv_lic2     = ifelse(drv_lic2     == 0, drv_lic1, drv_lic2))  
  
  
   
  # Doing averages whenever there's a secondary driver
  td$drv_age_avg = round((td$wgt_drv_age1 * td$drv_age1
                          + td$wgt_drv_age2 * td$drv_age2),digits=0)
  
  td$drv_lic_avg = round((td$wgt_drv_age1 * td$drv_lic1 
                          + td$wgt_drv_age2 * td$drv_lic2),digits=0)
  
  td$drv_nyl_avg = round((td$wgt_drv_age1 * td$drv_nyl1 
                          + td$wgt_drv_age2 * td$drv_nyl2),digits=0)
  
  
  td$drv_nyl1_o_age = td$drv_nyl1 / td$drv_age1
  td$drv_nyl2_o_age = td$drv_nyl2 / td$drv_age2
  #td <- td %>% mutate(drv_nyl2_o_age = ifelse(is.nan(drv_nyl2_o_age),NA,drv_nyl2_o_age))
  
  td$drv_lic1_o_age = td$drv_lic1 / td$drv_age1 
  td$drv_lic2_o_age = td$drv_lic2 / td$drv_age2
  #td <- td %>% mutate(drv_lic2_o_age = ifelse(is.nan(drv_lic2_o_age),NA,drv_lic2_o_age))  
  
  td$drv_age_p_nyl1    = td$drv_age1 + td$drv_nyl1  #a proxy of "global" experience
  td$drv_age_p_nyl2    = td$drv_age2 + td$drv_nyl2
  
  td$drv_age_p_lic1    = td$drv_age1 + td$drv_lic1 
  td$drv_age_p_lic2    = td$drv_age2 + td$drv_lic2
  
  td$drv_age_p_nyl1_x_pc    = td$drv_age_p_nyl1 * td$pc
  #td$drv_age_p_nyl1_x_pco   = td$drv_age_p_nyl1 * td$pco  
  td$drv_age_p_nyl1_x_pu    = td$drv_age_p_nyl1 * td$pu 
  td$drv_age_p_nyl1_x_pcpu  = td$drv_age_p_nyl1 * td$pcpu
  td$drv_age_p_nyl1_x_pcppu = td$drv_age_p_nyl1 * td$pcppu
  
  td$drv_age_p_lic1_x_pc    = td$drv_age_p_lic1 * td$pc
  #td$drv_age_p_lic1_x_pco   = td$drv_age_p_lic1 * td$pco  
  td$drv_age_p_lic1_x_pu    = td$drv_age_p_lic1 * td$pu 
  td$drv_age_p_lic1_x_pcpu  = td$drv_age_p_lic1 * td$pcpu
  td$drv_age_p_lic1_x_pcppu = td$drv_age_p_lic1 * td$pcppu
  
  
  #averages of the previous features
  td$drv_nyl_o_age_avg =  td$drv_nyl_avg / td$drv_age_avg
  
  td$drv_lic_o_age_avg =  td$drv_lic_avg / td$drv_age_avg
  
  td$drv_age_p_nyl_avg    =  round((td$wgt_drv_age1 * td$drv_age_p_nyl1
                                    + td$wgt_drv_age2 * td$drv_age_p_nyl2),digits=0)
  td$drv_age_p_lic_avg    =  round((td$wgt_drv_age1 * td$drv_age_p_lic1
                                    + td$wgt_drv_age2 * td$drv_age_p_lic2),digits=0)      
  
  
  # Proportion of time Licensed that the driver has been client with the ins company.
  td$pol_duration_o_age = td$pol_duration / td$drv_age1
  td$pol_duration_o_nyl = td$pol_duration / td$drv_nyl1
  td$pol_duration_o_lic = td$pol_duration / td$drv_lic1  
  
  td$pol_duration_o_age_x_pc    = td$pol_duration_o_age * td$pc
  #td$pol_duration_o_age_x_pco   = td$pol_duration_o_age * td$pco
  td$pol_duration_o_age_x_pu    = td$pol_duration_o_age * td$pu
  td$pol_duration_o_age_x_pcpu  = td$pol_duration_o_age * td$pcpu
  td$pol_duration_o_age_x_pcppu = td$pol_duration_o_age * td$pcppu 
  
  td$pol_duration_o_nyl_x_pc    = td$pol_duration_o_nyl * td$pc
  #td$pol_duration_o_nyl_x_pco   = td$pol_duration_o_nyl * td$pco
  td$pol_duration_o_nyl_x_pu    = td$pol_duration_o_nyl * td$pu
  td$pol_duration_o_nyl_x_pcpu  = td$pol_duration_o_nyl * td$pcpu
  td$pol_duration_o_nyl_x_pcppu = td$pol_duration_o_nyl * td$pcppu
  
  td$pol_duration_o_lic_x_pc    = td$pol_duration_o_lic * td$pc
  #td$pol_duration_o_lic_x_pco   = td$pol_duration_o_lic * td$pco
  td$pol_duration_o_lic_x_pu    = td$pol_duration_o_lic * td$pu
  td$pol_duration_o_lic_x_pcpu  = td$pol_duration_o_lic * td$pcpu
  td$pol_duration_o_lic_x_pcppu = td$pol_duration_o_lic * td$pcppu
  
  # No Claims Discount
  td <- td %>% mutate(NCD_at_zero = ifelse(pol_no_claims_discount==0,1,0))
  
  # I think multiplying NYL x NCD could give an interesting proxy of risk.
  td$ncd_x_age           = td$pol_no_claims_discount * td$drv_age1
  td$ncd_x_lnage         = td$pol_no_claims_discount * log(td$drv_age1)  
  td$ncd_x_nyl           = td$pol_no_claims_discount * td$drv_nyl1
  td$ncd_x_lnnyl         = td$pol_no_claims_discount * log(td$drv_nyl1)
  td$lnnyl_p_ncd_x_lnnyl = log(td$drv_nyl1) + (td$pol_no_claims_discount * log(td$drv_nyl1)) 
  td$lnage_p_ncd_x_lnage = log(td$drv_age1) + (td$pol_no_claims_discount * log(td$drv_age1))  
  td$ncd_x_age_p_nyl1    = td$pol_no_claims_discount * td$drv_age_p_nyl1
  
  td$ncd_x_age_x_pc            = td$ncd_x_age * td$pc
  td$lnage_p_ncd_x_lnage_x_pc  = td$lnage_p_ncd_x_lnage * td$pc
  td$ncd_x_age_p_nyl1_x_pc     = td$ncd_x_age_p_nyl1 * td$pc
  
  #td$ncd_x_age_x_pco           = td$ncd_x_age * td$pco
  #td$lnage_p_ncd_x_lnage_x_pco = td$lnage_p_ncd_x_lnage * td$pco
  #td$ncd_x_age_p_nyl1_x_pco    = td$ncd_x_age_p_nyl1 * td$pco
  
  td$ncd_x_age_x_pu            = td$ncd_x_age * td$pu
  td$lnage_p_ncd_x_lnage_x_pu  = td$lnage_p_ncd_x_lnage * td$pu
  td$ncd_x_age_p_nyl1_x_pu     = td$ncd_x_age_p_nyl1 * td$pu
  
  td$ncd_x_age_x_pcpu           = td$ncd_x_age * td$pcpu
  td$lnage_p_ncd_x_lnage_x_pcpu = td$lnage_p_ncd_x_lnage * td$pcpu
  td$ncd_x_age_p_nyl1_x_pcpu    = td$ncd_x_age_p_nyl1 * td$pcpu
  
  td$ncd_x_age_x_pcppu           = td$ncd_x_age * td$pcppu
  td$lnage_p_ncd_x_lnage_x_pcppu = td$lnage_p_ncd_x_lnage * td$pcppu
  td$ncd_x_age_p_nyl1_x_pcppu    = td$ncd_x_age_p_nyl1 * td$pcppu
  
  #would it be worthful to make those variables in terms of average? maybe, let's see if the more basic variable stand out
  
  
 
  
  # Adding a flag for presence of a young male driver
  td <- td %>% mutate(flag_young_male = ifelse((drv_sex1 == "M" & drv_age1 < 25) | (drv_sex2 == "M" & drv_age2 < 25),1,0))
  td$flag_young_male <- as.factor(td$flag_young_male)  
  
  # Adding a flag for presence of a young male driver
  td <- td %>% mutate(flag_veryyoung_male = ifelse((drv_sex1 == "M" & drv_age1 < 22) | (drv_sex2 == "M" & drv_age2 < 22),1,0))
  td$flag_veryyoung_male <- as.factor(td$flag_veryyoung_male)    
  
  # Adding a flag for presence of a senior
  td <- td %>% mutate(flag_senior = ifelse(drv_age1 >= 85,1,0))
  td$flag_senior <- as.factor(td$flag_senior)  
  
  # Adding indicators of relationship between drivers
  td$age_gap = td$drv_age1 - td$drv_age2
  td <- td %>% mutate(age_gap_gr = case_when(age_gap <= 17 ~ "gap_low_17",
                                             age_gap >  17 ~ "gap_high_17",
                                             TRUE ~ "no_occ_drv"))
  td$age_gap_gr <- as.factor(td$age_gap_gr) 
  
  td <- td %>% mutate(age_gap_gr2 = case_when(age_gap <= -17 ~ "senior_as_occ",
                                              (age_gap > -17 & age_gap <= 17) ~ "couple",
                                              age_gap > 17 ~ "kid_as_occ",
                                              TRUE ~ "no_occ_drv"))
  
  td$age_gap_gr2 <- as.factor(td$age_gap_gr2) 
  
  
  
  
  # dropping secondary driver
  #td <- td %>% dplyr::select (-c(drv_age2, drv_age_lic2))
  #i would drop it under a GLM model that can't handle NAs, however, for xgboost, i will keep it.
  
  
  
  # assigning mean
  
  td <- td %>% mutate(vh_age = case_when(is.na(vh_age) ~mean(vh_age, na.rm=TRUE),
                                         TRUE ~as.numeric(vh_age)
  )
  )
  
  td <- td %>% mutate(vh_speed = case_when(is.na(vh_speed) ~mean(vh_speed, na.rm=TRUE),
                                           TRUE ~as.numeric(vh_speed)
  )
  )
  
  # td <- td %>% mutate(vh_weight = case_when(is.na(vh_weight) ~mean(vh_weight, na.rm=TRUE),
  #                                                                       TRUE ~as.numeric(vh_weight)
  # )
  # )
  # 
  td <- td %>% mutate(vh_value = case_when(is.na(vh_value) ~mean(vh_value, na.rm=TRUE),
                                           TRUE ~as.numeric(vh_value)
  )
  )
  
  td <- add_vh_weight(td)
  
  
  # Because I think vehicle value might be at year 1, I will make a depreciated indicator
  # 20% first year, then 15% onwards. Going back up after 30 years,  collection cars....
  
  td <- td %>% mutate(depreciation = case_when( vh_age <= 1               ~   1 ,
                                                vh_age == 2               ~ (1-0.20),
                                                (vh_age >= 3  & vh_age<=20)~ (1-0.20)*((1-0.15)^(vh_age-2)),
                                                (vh_age >  20 & vh_age<=30)~ (1-0.20)*((1-0.15)^(  20  -2)),
                                                vh_age >  30              ~ (1-0.20)*((1-0.15)^(  20  -2))*(1.10^(vh_age-30)),
                                                TRUE ~ 1))
  
  td$vh_valuedep = td$vh_value * td$depreciation
  
  
  # 
  names(td)[sapply(td, anyNA)]
  # ok no more na! yay!
  
  #td$claim_amount <- factor(td$claim_amount)
  #str(td)
  
  # Some population have 0s... I don't believe it. I will try to impute a value.
  # Need to adjust populations that are recorded as zeros.
  # I give up, I will simply fix them to 1.
  td <- td %>% mutate(population = ifelse(population == 0, 1, population))
  td <- td %>% mutate(population = ifelse(is.na(population), 1, population))
  
  # Because I'm doing Log transforms, Safety nets in case we have 0s or NAs
  td <- td %>% mutate(town_surface_area = ifelse(town_surface_area == 0, 1, town_surface_area)) 
  td <- td %>% mutate(town_surface_area = ifelse(is.na(town_surface_area), 1, town_surface_area))
  
  td <- td %>% mutate(vh_age = ifelse(vh_age == 0, 1, vh_age))  
  td <- td %>% mutate(vh_speed = ifelse(vh_speed == 0, 1, vh_speed))  
  td <- td %>% mutate(vh_value = ifelse(vh_value == 0, 1, vh_value))   
  td <- td %>% mutate(vh_weight = ifelse(vh_weight == 0, 1, vh_weight)) 
  
  
  
  # Adding Population Density
  td$pop_density         = td$population / td$town_surface_area
  td$pop_density_x_pc    = td$pop_density * td$pc
  #td$pop_density_x_pco   = td$pop_density * td$pco
  td$pop_density_x_pu    = td$pop_density * td$pu
  td$pop_density_x_pcpu  = td$pop_density * td$pcpu  
  td$pop_density_x_pcppu = td$pop_density * td$pcppu
  
  # Would a yound driver be more risk in a more densified city? Probably...
  td$drv_age_o_population = td$drv_age1 / td$population
  td$drv_age_o_population_avg = td$drv_age_avg / td$population 
  
  td$drv_age_o_population_x_pc        = td$drv_age_o_population_avg * td$pc
  #td$drv_age_o_population_x_pco       = td$drv_age_o_population_avg * td$pco         
  td$drv_age_o_population_x_pu        = td$drv_age_o_population_avg * td$pu
  td$drv_age_o_population_x_pcpu      = td$drv_age_o_population_avg * td$pcpu      
  td$drv_age_o_population_x_pcppu     = td$drv_age_o_population_avg * td$pcppu 
  
  td$drv_age_o_population_avg_x_pc    = td$drv_age_o_population_avg * td$pc
  #td$drv_age_o_population_avg_x_pco   = td$drv_age_o_population_avg * td$pco         
  td$drv_age_o_population_avg_x_pu    = td$drv_age_o_population_avg * td$pu
  td$drv_age_o_population_avg_x_pcpu  = td$drv_age_o_population_avg * td$pcpu
  td$drv_age_o_population_avg_x_pcppu = td$drv_age_o_population_avg * td$pcppu
  
  td$drv_age_o_density  = td$drv_age1 / td$pop_density
  td$drv_age_o_density_avg  = td$drv_age_avg / td$pop_density
  
  td$drv_nyl_o_population = td$drv_nyl1 / td$population
  td$drv_nyl_o_population_avg = td$drv_nyl_avg / td$population  
  td$drv_nyl_o_population_x_pc    = td$drv_nyl_o_population * td$pc
  #td$drv_nyl_o_population_x_pco   = td$drv_nyl_o_population * td$pco         
  td$drv_nyl_o_population_x_pu    = td$drv_nyl_o_population * td$pu
  td$drv_nyl_o_population_x_pcpu  = td$drv_nyl_o_population * td$pcpu 
  td$drv_nyl_o_population_x_pcppu = td$drv_nyl_o_population * td$pcppu
  
  td$drv_nyl_o_density  = td$drv_nyl1 / td$pop_density
  td$drv_nyl_o_density_avg  = td$drv_nyl_avg / td$pop_density 
  
  
  # Value of vehicle in relation to territoy
  td$vh_value_o_population = td$vh_value / td$population
  td$vh_value_o_town       = td$vh_value / td$town_surface_area  
  td$vh_value_o_density    = td$vh_value / td$pop_density
  
  td$vh_valuedep_o_population = td$vh_valuedep / td$population
  td$vh_valuedep_o_town       = td$vh_valuedep / td$town_surface_area  
  td$vh_valuedep_o_density    = td$vh_valuedep / td$pop_density
  
  td$vh_speed_o_density    = td$vh_speed / td$pop_density 
  td$vh_speed_o_town       = td$vh_speed / td$town_surface_area   
  
  td$vh_value_o_town_x_pc    = td$vh_value_o_town * td$pc
  #td$vh_value_o_town_x_pco   = td$vh_value_o_town * td$pco
  td$vh_value_o_town_x_pu    = td$vh_value_o_town * td$pu
  td$vh_value_o_town_x_pcpu  = td$vh_value_o_town * td$pcpu
  td$vh_value_o_town_x_pcppu = td$vh_value_o_town * td$pcppu
  
  td$vh_valuedep_o_town_x_pc    = td$vh_valuedep_o_town * td$pc
  #td$vh_valuedep_o_town_x_pco   = td$vh_valuedep_o_town * td$pco
  td$vh_valuedep_o_town_x_pu    = td$vh_valuedep_o_town * td$pu
  td$vh_valuedep_o_town_x_pcpu  = td$vh_valuedep_o_town * td$pcpu
  td$vh_valuedep_o_town_x_pcppu = td$vh_valuedep_o_town * td$pcppu
  
  td$vh_value_o_density_x_pc    = td$vh_value_o_density * td$pc
  #td$vh_value_o_density_x_pco   = td$vh_value_o_density * td$pco  
  td$vh_value_o_density_x_pu    = td$vh_value_o_density * td$pu
  td$vh_value_o_density_x_pcpu  = td$vh_value_o_density * td$pcpu
  td$vh_value_o_density_x_pcppu = td$vh_value_o_density * td$pcppu
  
  td$vh_valuedep_o_density_x_pc    = td$vh_valuedep_o_density * td$pc
  #td$vh_valuedep_o_density_x_pco   = td$vh_valuedep_o_density * td$pco  
  td$vh_valuedep_o_density_x_pu    = td$vh_valuedep_o_density * td$pu
  td$vh_valuedep_o_density_x_pcpu  = td$vh_valuedep_o_density * td$pcpu
  td$vh_valuedep_o_density_x_pcppu = td$vh_valuedep_o_density * td$pcppu 
  
  
  td$vh_speed_o_density_x_pc    = td$vh_speed_o_density * td$pc
  #td$vh_speed_o_density_x_pco   = td$vh_speed_o_density * td$pco  
  td$vh_speed_o_density_x_pu    = td$vh_speed_o_density * td$pu
  td$vh_speed_o_density_x_pcpu  = td$vh_speed_o_density * td$pcpu  
  td$vh_speed_o_density_x_pcppu = td$vh_speed_o_density * td$pcppu
  
  td$vh_speed_o_town_x_pc    = td$vh_speed_o_town * td$pc
  #td$vh_speed_o_town_x_pco   = td$vh_speed_o_town * td$pco
  td$vh_speed_o_town_x_pu    = td$vh_speed_o_town * td$pu
  td$vh_speed_o_town_x_pcpu  = td$vh_speed_o_town * td$pcpu
  td$vh_speed_o_town_x_pcppu = td$vh_speed_o_town * td$pcppu
  
  # Should I make territory groups?
  # Find unique combinations of population x town surface, create groups and do target encoding.
  
  
  # Vehicle information variables
  td$vh_speed_o_weight                  = td$vh_speed / td$vh_weight
  td$vh_value_o_weight                  = td$vh_value / td$vh_weight
  td$vh_lnspeed_x_lnvalue               = log(td$vh_speed) * log(td$vh_value)  #a combination of both leads to much more risk!
  td$vh_lnspeed_x_lnvaluedep            = log(td$vh_speed) * log(td$vh_valuedep)  #a combination of both leads to much more risk!
  td$vh_lnspeed_x_lnweight              = log(td$vh_speed) * log(td$vh_weight)  #a combination of both leads to much more risk as it can do much damage
  #td$vh_lnspeed_o_lnweight             = log(td$vh_speed) / log(td$vh_weight)  
  td$vh_lnspeed_x_lnvalue_x_lnweight    = log(td$vh_speed) * log(td$vh_value) * log(td$vh_weight)
  td$vh_lnspeed_x_lnvaluedep_x_lnweight = log(td$vh_speed) * log(td$vh_valuedep) * log(td$vh_weight)
  td$vh_lnspeed_x_lnvalue_o_lnweight    = log(td$vh_speed) * log(td$vh_value) / log(td$vh_weight) 
  td$vh_lnspeed_x_lnvaluedep_o_lnweight = log(td$vh_speed) * log(td$vh_valuedep) / log(td$vh_weight)
  
  td$vh_lnvalue_o_vh_age                = log(td$vh_value) / td$vh_age  #the higher the more risk?
  td$vh_lnvaluedep_o_vh_age             = log(td$vh_valuedep) / td$vh_age  #the higher the more risk?
  td$vh_lnvalue_x_lnspeed_o_vh_age      = log(td$vh_value) * log(td$vh_speed) / td$vh_age  #the higher the more risk?
  td$vh_lnvaluedep_x_lnspeed_o_vh_age   = log(td$vh_valuedep) * log(td$vh_speed) / td$vh_age  #the higher the more risk?
  
  td$vh_lnvalue_o_vh_age_x_pc                 = td$vh_lnvalue_o_vh_age * td$pc 
  td$vh_lnvaluedep_o_vh_age_x_pc              = td$vh_lnvaluedep_o_vh_age * td$pc 
  td$vh_lnvalue_x_lnspeed_o_vh_age_x_pc       = td$vh_lnvalue_x_lnspeed_o_vh_age * td$pc
  td$vh_lnvaluedep_x_lnspeed_o_vh_age_x_pc    = td$vh_lnvaluedep_x_lnspeed_o_vh_age * td$pc
  td$pc_o_vh_age                              = td$pc / td$vh_age
  td$vh_lnspeed_x_lnweight_x_pc               = td$vh_lnspeed_x_lnweight * td$pc
  td$vh_lnspeed_x_lnvalue_x_lnweight_x_pc     = td$vh_lnspeed_x_lnvalue_x_lnweight * td$pc
  td$vh_lnspeed_x_lnvaluedep_x_lnweight_x_pc  = td$vh_lnspeed_x_lnvaluedep_x_lnweight * td$pc
  td$vh_lnvalue_x_pc                          = log(td$vh_value) * td$pc
  td$vh_lnvaluedep_x_pc                       = log(td$vh_valuedep) * td$pc
  
  #td$vh_lnvalue_o_vh_age_x_pco                = td$vh_lnvalue_o_vh_age * td$pco 
  #td$vh_lnvaluedep_o_vh_age_x_pco             = td$vh_lnvaluedep_o_vh_age * td$pco 
  #td$vh_lnvalue_x_lnspeed_o_vh_age_x_pco      = td$vh_lnvalue_x_lnspeed_o_vh_age * td$pco
  #td$vh_lnvaluedep_x_lnspeed_o_vh_age_x_pco   = td$vh_lnvaluedep_x_lnspeed_o_vh_age * td$pco
  #td$pco_o_vh_age                             = td$pco / td$vh_age
  #td$vh_lnspeed_x_lnweight_x_pco              = td$vh_lnspeed_x_lnweight * td$pco
  #td$vh_lnspeed_x_lnvalue_x_lnweight_x_pco    = td$vh_lnspeed_x_lnvalue_x_lnweight * td$pco
  #td$vh_lnspeed_x_lnvaluedep_x_lnweight_x_pco = td$vh_lnspeed_x_lnvaluedep_x_lnweight * td$pco
  #td$vh_lnvalue_x_pco                         = log(td$vh_value) * td$pco
  #td$vh_lnvaluedep_x_pco                      = log(td$vh_valuedep) * td$pco
  
  
  td$vh_lnvalue_o_vh_age_x_pu                 = td$vh_lnvalue_o_vh_age * td$pu 
  td$vh_lnvaluedep_o_vh_age_x_pu              = td$vh_lnvaluedep_o_vh_age * td$pu 
  td$vh_lnvalue_x_lnspeed_o_vh_age_x_pu       = td$vh_lnvalue_x_lnspeed_o_vh_age * td$pu
  td$vh_lnvaluedep_x_lnspeed_o_vh_age_x_pu    = td$vh_lnvaluedep_x_lnspeed_o_vh_age * td$pu
  td$pu_o_vh_age                              = td$pu / td$vh_age
  td$vh_lnspeed_x_lnweight_x_pu               = td$vh_lnspeed_x_lnweight * td$pu
  td$vh_lnspeed_x_lnvalue_x_lnweight_x_pu     = td$vh_lnspeed_x_lnvalue_x_lnweight * td$pu
  td$vh_lnspeed_x_lnvaluedep_x_lnweight_x_pu  = td$vh_lnspeed_x_lnvaluedep_x_lnweight * td$pu
  td$vh_lnvalue_x_pu                          = log(td$vh_value) * td$pu
  td$vh_lnvaluedep_x_pu                       = log(td$vh_valuedep) * td$pu  
  
  td$vh_lnvalue_x_lnspeed_o_vh_age_x_pcpu      = td$vh_lnvalue_x_lnspeed_o_vh_age * td$pcpu
  td$vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcpu   = td$vh_lnvaluedep_x_lnspeed_o_vh_age * td$pcpu      
  td$vh_lnvalue_o_vh_age_x_pcpu                = td$vh_lnvalue_o_vh_age * td$pcpu
  td$vh_lnvaluedep_o_vh_age_x_pcpu             = td$vh_lnvaluedep_o_vh_age * td$pcpu      
  td$vh_lnspeed_x_lnweight_x_pcpu              = td$vh_lnspeed_x_lnweight * td$pcpu
  td$pcpu_o_vh_age                             = td$pcpu / td$vh_age
  td$vh_lnspeed_x_lnvalue_x_lnweight_x_pcpu    = td$vh_lnspeed_x_lnvalue_x_lnweight * td$pcpu
  td$vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcpu = td$vh_lnspeed_x_lnvaluedep_x_lnweight * td$pcpu
  td$vh_lnvalue_x_pcpu                         = log(td$vh_value) * td$pcpu
  td$vh_lnvaluedep_x_pcpu                      = log(td$vh_valuedep) * td$pcpu
  
  td$vh_weight_x_pc                         = td$vh_weight * td$pc
  #td$vh_weight_x_pco                        = td$vh_weight * td$pco
  td$vh_weight_x_pu                         = td$vh_weight * td$pu
  td$vh_weight_x_pcpu                       = td$vh_weight * td$pcpu
  
  
  td$vh_lnvalue_x_lnspeed_o_vh_age_x_pcppu      = td$vh_lnvalue_x_lnspeed_o_vh_age * td$pcppu
  td$vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcppu   = td$vh_lnvaluedep_x_lnspeed_o_vh_age * td$pcppu      
  td$vh_lnvalue_o_vh_age_x_pcppu                = td$vh_lnvalue_o_vh_age * td$pcppu
  td$vh_lnvaluedep_o_vh_age_x_pcppu             = td$vh_lnvaluedep_o_vh_age * td$pcppu      
  td$vh_lnspeed_x_lnweight_x_pcppu              = td$vh_lnspeed_x_lnweight * td$pcppu
  td$pcppu_o_vh_age                             = td$pcppu / td$vh_age
  td$vh_lnspeed_x_lnvalue_x_lnweight_x_pcppu    = td$vh_lnspeed_x_lnvalue_x_lnweight * td$pcppu
  td$vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcppu = td$vh_lnspeed_x_lnvaluedep_x_lnweight * td$pcppu
  td$vh_lnvalue_x_pcppu                         = log(td$vh_value) * td$pcppu
  td$vh_lnvaluedep_x_pcppu                      = log(td$vh_valuedep) * td$pcppu  
  
  td$vh_weight_x_pcppu                          = td$vh_weight * td$pcppu
  
  #Relationship between driver and vehicle 
  # assumption: If a young driver drives a faster vehicle, it can lead to more risk
  # assumption: If a young driver drives a heavier vehicle, it can lead to more risk (?)
  td$vh_speed_o_drv_age      =  td$vh_speed / td$drv_age1
  td$vh_weight_o_drv_age     =  td$vh_weight / td$drv_age1
  td$vh_lnvalue_o_drv_age    =  log(td$vh_value) / td$drv_age1
  td$vh_lnvaluedep_o_drv_age =  log(td$vh_valuedep) / td$drv_age1  
  td$vh_speed_o_drv_nyl      =  td$vh_speed / td$drv_nyl1
  td$vh_weight_o_drv_nyl     =  td$vh_weight / td$drv_nyl1
  td$vh_lnvalue_o_drv_nyl    =  log(td$vh_value) / td$drv_nyl1  
  td$vh_lnvaluedep_o_drv_nyl =  log(td$vh_valuedep) / td$drv_nyl1
  
  td$vh_lnvalue_o_drv_age_x_pc    = td$vh_lnvalue_o_drv_age * td$pc
  #td$vh_lnvalue_o_drv_age_x_pco   = td$vh_lnvalue_o_drv_age * td$pco    
  td$vh_lnvalue_o_drv_age_x_pu    = td$vh_lnvalue_o_drv_age * td$pu
  td$vh_lnvalue_o_drv_age_x_pcpu  = td$vh_lnvalue_o_drv_age * td$pcpu
  td$vh_lnvalue_o_drv_age_x_pcppu = td$vh_lnvalue_o_drv_age * td$pcppu
  
  td$vh_lnvaluedep_o_drv_age_x_pc    = td$vh_lnvaluedep_o_drv_age * td$pc
  #td$vh_lnvaluedep_o_drv_age_x_pco   = td$vh_lnvaluedep_o_drv_age * td$pco    
  td$vh_lnvaluedep_o_drv_age_x_pu    = td$vh_lnvaluedep_o_drv_age * td$pu
  td$vh_lnvaluedep_o_drv_age_x_pcpu  = td$vh_lnvaluedep_o_drv_age * td$pcpu
  td$vh_lnvaluedep_o_drv_age_x_pcppu = td$vh_lnvaluedep_o_drv_age * td$pcppu    
  
  td$vh_age_p_drv_age       =  td$vh_age + td$drv_age1  # if both are small... more risk!
  td$vh_age_p_drv_nyl       =  td$vh_age + td$drv_nyl1
  td$vh_age_p_drv_nyl_x_ncd =  (td$vh_age + td$drv_nyl1) * td$pol_no_claims_discount
  td$vh_age_o_drv_nyl       =  td$vh_age / td$drv_nyl1
  td$vh_age_o_drv_nyl_x_pc  =  td$vh_age_o_drv_nyl * td$pc
  #td$vh_age_o_drv_nyl_x_pco =  td$vh_age_o_drv_nyl * td$pco
  td$vh_age_o_drv_nyl_x_pu  =  td$vh_age_o_drv_nyl * td$pu
  
  td$vh_age_p_drv_nyl_x_ncd_x_pc    = td$vh_age_p_drv_nyl_x_ncd * td$pc
  #td$vh_age_p_drv_nyl_x_ncd_x_pco   = td$vh_age_p_drv_nyl_x_ncd * td$pco  
  td$vh_age_p_drv_nyl_x_ncd_x_pu    = td$vh_age_p_drv_nyl_x_ncd * td$pu
  td$vh_age_p_drv_nyl_x_ncd_x_pcpu  = td$vh_age_p_drv_nyl_x_ncd * td$pcpu
  td$vh_age_p_drv_nyl_x_ncd_x_pcppu = td$vh_age_p_drv_nyl_x_ncd * td$pcppu
  
  td$vh_age_o_drv_nyl_x_pcpu        = td$vh_age_o_drv_nyl* td$pcpu
  td$vh_age_o_drv_nyl_x_pcppu       = td$vh_age_o_drv_nyl* td$pcppu
  
  # Relationship between ncd, driver, nyl and density, and vhmm
  # will add some if density omces uup.
  
  #"id_policy",
  drops <- c("id_policy","claim_cnt", 
             "claim_amount_cap", "pred_glm",
             "wgt_drv_age1", "wgt_drv_age2", 
             "drv_sex2", 
             "drv_age_lic1","drv_age_lic2")
  
  
  td <- td[ , !(names(td) %in% drops)]
  
  #str(td)
  
  
  td <- add_vh_groups(td)
  td <- add_vh_order(td)  
  td <- add_vh_popularity(td) 
  
  if (leakage_blocker == TRUE) {
    td$vh_make_model_enc = 1
    td$vh_mm_ord = 1
  }
    
  
  #td <- add_vh_order_train(td)  
  #td <- add_vh_popularity_train(td)  
  
  if (keep_vh_make_model == TRUE){
    td$vh_make_model <- as.factor(td$vh_make_model)
  } 
  else if (keep_vh_make_model == FALSE) {
    td <- td[ , !(names(td) %in% "vh_make_model")]
  }

  
  td$vh_mm_o_drv_nyl      = td$vh_make_model_enc / td$drv_nyl1
  td$vh_mm_x_lnspeed      = td$vh_make_model_enc * log(td$vh_speed)
  td$vh_mm_x_lnvalue      = td$vh_make_model_enc * log(td$vh_value)
  td$vh_mm_x_lnvaluedep   = td$vh_make_model_enc * log(td$vh_valuedep)
  
  td$vh_mm_o_drv_nyl_x_pc     = td$vh_mm_o_drv_nyl * td$pc
  td$vh_mm_x_lnspeed_x_pc     = td$vh_mm_x_lnspeed * td$pc
  td$vh_mm_x_lnvalue_x_pc     = td$vh_mm_x_lnvalue * td$pc
  td$vh_mm_x_lnvaluedep_x_pc  = td$vh_mm_x_lnvaluedep * td$pc
  
  #td$vh_mm_o_drv_nyl_x_pco    = td$vh_mm_o_drv_nyl * td$pco
  #td$vh_mm_x_lnspeed_x_pco    = td$vh_mm_x_lnspeed * td$pco
  #td$vh_mm_x_lnvalue_x_pco    = td$vh_mm_x_lnvalue * td$pco
  #td$vh_mm_x_lnvaluedep_x_pco = td$vh_mm_x_lnvaluedep * td$pco  
  
  td$vh_mm_o_drv_nyl_x_pu    = td$vh_mm_o_drv_nyl * td$pu
  td$vh_mm_x_lnspeed_x_pu    = td$vh_mm_x_lnspeed * td$pu
  td$vh_mm_x_lnvalue_x_pu    = td$vh_mm_x_lnvalue * td$pu  
  td$vh_mm_x_lnvaluedep_x_pu = td$vh_mm_x_lnvaluedep * td$pu  
  
  td$vh_mm_o_drv_nyl_x_pcpu     = td$vh_mm_o_drv_nyl * td$pcpu
  td$vh_mm_x_lnspeed_x_pcpu     = td$vh_mm_x_lnspeed * td$pcpu
  td$vh_mm_x_lnvalue_x_pcpu     = td$vh_mm_x_lnvalue * td$pcpu
  td$vh_mm_x_lnvaluedep_x_pcpu  = td$vh_mm_x_lnvaluedep * td$pcpu
  
  td$vh_mm_o_drv_nyl_x_pcppu    = td$vh_mm_o_drv_nyl * td$pcppu
  td$vh_mm_x_lnspeed_x_pcppu    = td$vh_mm_x_lnspeed * td$pcppu
  td$vh_mm_x_lnvalue_x_pcppu    = td$vh_mm_x_lnvalue * td$pcppu
  td$vh_mm_x_lnvaluedep_x_pcppu = td$vh_mm_x_lnvaluedep * td$pcppu
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++
  #Added Janv 29, 2021
  
  # vh_mm_pop => the higher, the more "popular"
  # vh_mm_ord => the higher, the more accident claims it has
  td$vh_mmpop_x_mmord   = td$vh_mm_pop * td$vh_mm_ord
  td$vh_mm_x_mmpop   = td$vh_make_model_enc * td$vh_mm_pop
  td$vh_mm_x_mmord   = td$vh_make_model_enc * td$vh_mm_ord
  
  td$vh_mmpop_x_drv_nyl      = td$vh_mm_pop * td$drv_nyl1
  #td$vh_mmpop_x_lnspeed      = td$vh_mm_pop * log(td$vh_speed)
  #td$vh_mmpop_x_lnvalue      = td$vh_mm_pop * log(td$vh_value)
  #td$vh_mmpop_x_lnvaluedep   = td$vh_mm_pop * log(td$vh_valuedep)

  td$vh_lnspeed_d_mmpop      = log(td$vh_speed) / td$vh_mm_pop   
  td$vh_lnvalue_d_mmpop      = log(td$vh_value) / td$vh_mm_pop 
  td$vh_lnvaluedep_d_mmpop   = log(td$vh_valuedep) / td$vh_mm_pop 
  
  # td$vh_mmpop_x_drv_nyl_x_pc     = td$vh_mmpop_x_drv_nyl * td$pc
  # td$vh_mmpop_x_lnspeed_x_pc     = td$vh_mmpop_x_lnspeed * td$pc
  # td$vh_mmpop_x_lnvalue_x_pc     = td$vh_mmpop_x_lnvalue * td$pc
  # td$vh_mmpop_x_lnvaluedep_x_pc  = td$vh_mmpop_x_lnvaluedep * td$pc
  # 
  # td$vh_mmpop_x_drv_nyl_x_pco     = td$vh_mmpop_x_drv_nyl * td$pco
  # td$vh_mmpop_x_lnspeed_x_pco     = td$vh_mmpop_x_lnspeed * td$pco
  # td$vh_mmpop_x_lnvalue_x_pco     = td$vh_mmpop_x_lnvalue * td$pco
  # td$vh_mmpop_x_lnvaluedep_x_pco  = td$vh_mmpop_x_lnvaluedep * td$pco
  # 
  # td$vh_mmpop_x_drv_nyl_x_pu     = td$vh_mmpop_x_drv_nyl * td$pu
  # td$vh_mmpop_x_lnspeed_x_pu     = td$vh_mmpop_x_lnspeed * td$pu
  # td$vh_mmpop_x_lnvalue_x_pu     = td$vh_mmpop_x_lnvalue * td$pu
  # td$vh_mmpop_x_lnvaluedep_x_pu  = td$vh_mmpop_x_lnvaluedep * td$pu
  # 
  # td$vh_mmpop_x_drv_nyl_x_pcpu     = td$vh_mmpop_x_drv_nyl * td$pcpu 
  # td$vh_mmpop_x_lnspeed_x_pcpu      = td$vh_mmpop_x_lnspeed * td$pcpu 
  # td$vh_mmpop_x_lnvalue_x_pcpu      = td$vh_mmpop_x_lnvalue * td$pcpu 
  # td$vh_mmpop_x_lnvaluedep_x_pcpu   = td$vh_mmpop_x_lnvaluedep * td$pcpu 
  # 
  # td$vh_mmpop_x_drv_nyl_x_pcppu     = td$vh_mmpop_x_drv_nyl * td$pcppu 
  # td$vh_mmpop_x_lnspeed_x_pcppu      = td$vh_mmpop_x_lnspeed * td$pcppu 
  # td$vh_mmpop_x_lnvalue_x_pcppu      = td$vh_mmpop_x_lnvalue * td$pcppu 
  # td$vh_mmpop_x_lnvaluedep_x_pcppu   = td$vh_mmpop_x_lnvaluedep * td$pcppu 
  
  td$vh_drv_nyl_o_mmord      = td$drv_nyl1 / td$vh_mm_ord 
  td$vh_mmord_x_lnspeed      = td$vh_mm_ord * log(td$vh_speed)
  td$vh_mmord_x_lnvalue      = td$vh_mm_ord * log(td$vh_value)
  td$vh_mmord_x_lnvaluedep   = td$vh_mm_ord * log(td$vh_valuedep)
  
  
  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++
  td <-  add_driver_averages(td)
  td <-  add_nyl_averages(td)  
  

  
  td <- td[ , !(names(td)==c("ID"))]

  
  
  #for inflated data, I remove the year variable and the infl_factor
  #td <- td[ , !(names(td)=="age_gap")]
  td <- td[ , !(names(td)=="age_gap_gr2.no_occ_drv")]
  td <- td[ , !(names(td)=="drv_drv2.No")]
  td <- td[ , !(names(td)=="flag_senior.0")]
  td <- td[ , !(names(td)=="drv_sex1.M")] 
  
  
  td <- td %>% select(-avg_ncd,
                              -avg_nyl,
                              -avg_pco,
                              -avg_pol,
                              -avg_polsit,
                              -avg_pop,
                              -avg_vh_age,
                              -avg_vh_value,
                              -avg_vh_valuedep,
                              -Navg_age,
                              -Navg_ncd,
                              -Navg_pco,
                              -Navg_pol,
                              -Navg_polsit,
                              -Navg_vh_age,
                              -Navg_vh_value,
                              -Navg_vh_valuedep,
                              -NCD_at_zero,
                              -obs,
                              -depreciation
  )
  
  if (leakage_blocker == TRUE){
    
    td <- td %>% select(-vh_make_model_enc, 
                        -vh_mm_ord, 
                        -vh_mm_x_mmord)
  }
  
  cat("..... done \n")
  
  return(td)
}

preprocess_X_data <- function (x_raw, one_hot = TRUE, 
                               target_encoding = TRUE, 
                               leave_one_out = FALSE,
                               leakage_blocker = FALSE,
                               keep_vh_make_model = FALSE,
                               golden_feature = FALSE) {
 
  if (golden_feature == TRUE) {
    x_raw <- left_join(x_raw, f_golden, by = c("id_policy","year"))
    x_raw <- x_raw %>% mutate(F_claim_cnt     = replace_na(F_claim_cnt,0) ,
                              F_claim_cnt_c   = replace_na(F_claim_cnt_c,0) , 
                              F_claim_amt_c   = replace_na(F_claim_amt_c,0) , 
                              F_claim_amt_avg = replace_na(F_claim_amt_avg,0) , 
                              F_cov           = replace_na(F_cov,0) )
    
  }
  
  td <- prepreprocess_X_data(x_raw, leakage_blocker, keep_vh_make_model)
  
  cat(".Preprocess X data :: start ")
  
  if (one_hot == TRUE) {
    
    if (leave_one_out == FALSE){
      # One-Hot Encoding for TRAIN
      # Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
      #Dummy Variable Trap ... 
      #need to use fullRank=T  (for GLMs)
      #better to use fullRank=F for tree-based models
      dummies_model <- dummyVars(policy_cnt ~ ., data=td, fullRank=F)
      
      # Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
      training_mat <- predict(dummies_model, newdata = td)
      
      # # Convert to dataframe
      td <- data.frame(training_mat)
      
      # # See the structure of the new dataset
      #str(td)
    }
    else if (leave_one_out == TRUE){
      dummies_model <- dummyVars(policy_cnt ~ ., data=td, fullRank=T)
      # Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
      training_mat <- predict(dummies_model, newdata = td)
      
      # # Convert to dataframe
      td <- data.frame(training_mat)      
      
    }
  } 
  else {}
  
  if (target_encoding == TRUE) {}
  else {
    drop_TE <- c("pc","pu","vf","pcpu","pcppu")
    td <- td[, !(names(td) %in% drop_TE)]
  }
  
  
  cat(".... done \n")
  
  return(td) 
}





preprocess_Y_data <- function(y_raw){
  
  cat("Preprocess Y data :: start ")
  
  y_raw$claim_amount_inf = y_raw$claim_amount * y_raw$infl_factor

  #Capping claim amount
  y_raw <- y_raw %>% mutate(claim_amount_cap = ifelse(claim_amount>50000,50000,claim_amount))
  y_raw <- y_raw %>% mutate(claim_amount_inf_cap = ifelse(claim_amount_inf>50000,50000,claim_amount_inf))
  
  #Adding claim_cnt
  y_raw <- y_raw %>% mutate(claim_nb = ifelse(claim_amount > 0, 1, 0))
  y_raw <- y_raw %>% mutate(claim_occ = ifelse(claim_amount > 0, "Y", "N"))
  y_raw$claim_occ <- as.factor(y_raw$claim_occ)
  
  y_raw <- y_raw[ , !(names(y_raw) %in% c("year","infl_factor"))]

  # we will use claim_amount_inf_cap as severity response variable
  #y_clean <- y_raw$claim_amount_inf_cap
  
  #y_clean <- data.frame(y_raw$claim_amount_inf_cap,y_raw$claim_amount_cap)

  cat("... done \n")
  return(y_raw)
}



apply_cs_pca <- function(td, cs_model, pca_model){
  
    cat("+ Applying center, scale and pca + \n")
    cs_mat <- predict(cs_model, newdata = td)
    pca_mat <- predict(pca_model, newdata = cs_mat)
    
    td <- cbind(cs_mat,pca_mat)
    return(td)
}


trim_variables <- function(td){
  td <- td %>% select(vh_mm_x_lnvaluedep_x_pcpu,
                      vh_value_o_town_x_pcpu,
                      AS_drv_vh_valuedep,
                      ncd_x_age_p_nyl1_x_pc,
                      pol_pay_freq.Quarterly,
                      pol_duration_o_nyl_x_pcpu,
                      vh_fuel.Gasoline,
                      drv_nyl2_o_age,
                      vh_lnspeed_x_lnvalue_o_lnweight,
                      AS_drv_vh_age,
                      pop_density_x_pcpu,
                      pol_pay_freq.Yearly,
                      pol_sit_o_duration,
                      vh_speed_o_density_x_pcpu,
                      pol_ncd_occ_c,
                      pol_ncd_dec,
                      pol_payd.Yes,
                      AD_drv_polsit,
                      vh_speed_o_weight,
                      vh_lnvalue_o_drv_age_x_pu,
                      flag_young_male.1,
                      vh_type.Tourism,
                      drv_drv2.Yes,
                      vh_lnvaluedep_o_vh_age,
                      town_surface_area,
                      vh_lnvalue_o_drv_age_x_pcppu,
                      pol_duration_o_age_x_pc,
                      AD_drv_vh_valuedep,
                      vh_fuel.Hybrid,
                      vh_weight,
                      pol_ncd_inc_risk_c,
                      vh_speed_o_town_x_pc,
                      vh_mm_x_lnspeed,
                      flag_veryyoung_male.1,
                      drv_sex1.M,
                      population,
                      vh_lnvalue_o_drv_age_x_pcpu,
                      vh_mm_o_drv_nyl_x_pc,
                      vh_mm_x_lnvaluedep_x_pu,
                      vh_lnvalue_o_drv_age_x_pc,
                      pol_pay_freq.Monthly,
                      vh_valuedep_o_town,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_speed,
                      vh_lnspeed_x_lnvaluedep_o_lnweight,
                      vh_lnvalue_o_vh_age,
                      AS_nyl_ncd,
                      vh_lnvalue_o_drv_age_x_pc,
                      AD_drv_ncd,
                      AS_nyl_age,
                      flag_senior.1,
                      pol_coverage.Med1,
                      pol_ncd_dec_risk_c,
                      vh_lnvalue_x_pu,
                      age_gap_gr.gap_low_17,
                      vh_lnspeed_x_lnvalue,
                      vh_age,
                      vh_lnvaluedep_x_lnspeed_o_vh_age,
                      vh_lnvalue_x_lnspeed_o_vh_age,
                      pol_usage.WorkPrivate,
                      age_gap_gr2.senior_as_occ,
                      vh_mm_x_lnvaluedep,
                      vh_age_p_drv_nyl_x_ncd_x_pu,
                      vh_valuedep,
                      drv_lic2_o_age,
                      pol_ncd_inc_risk_ca,
                      age_gap_gr2.kid_as_occ,
                      pol_ncd_dec_risk_ca,
                      AD_nyl_pol,
                      vh_value_o_town,
                      vh_lnvaluedep_o_vh_age_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcpu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcppu,
                      vh_mm_x_lnvaluedep_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pcppu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcpu,
                      pcppu_o_vh_age,
                      vh_lnvaluedep_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcppu,
                      AS_nyl_vh_age,
                      vh_lnvalue_o_vh_age_x_pc,
                      lnage_p_ncd_x_lnage_x_pu,
                      vh_lnvaluedep_o_vh_age_x_pc,
                      vh_mm_x_lnvalue_x_pcppu,
                      vh_lnvalue_x_pc,
                      vh_lnvaluedep_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      AS_drv_polsit,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcppu,
                      vh_mmord_x_lnvaluedep,
                      vh_mmpop_x_mmord,
                      vh_mmord_x_lnvalue,
                      vh_lnvaluedep_d_mmpop,
                      starts_with("PC",ignore.case = FALSE ))
  return(td)
}


trim_variables_lb <- function(td){
  td <- td %>% select(vh_mm_x_lnvaluedep_x_pcpu,
                      vh_value_o_town_x_pcpu,
                      AS_drv_vh_valuedep,
                      ncd_x_age_p_nyl1_x_pc,
                      pol_pay_freq.Quarterly,
                      pol_duration_o_nyl_x_pcpu,
                      vh_fuel.Gasoline,
                      drv_nyl2_o_age,
                      vh_lnspeed_x_lnvalue_o_lnweight,
                      AS_drv_vh_age,
                      pop_density_x_pcpu,
                      pol_pay_freq.Yearly,
                      pol_sit_o_duration,
                      vh_speed_o_density_x_pcpu,
                      pol_ncd_occ_c,
                      pol_ncd_dec,
                      pol_payd.Yes,
                      AD_drv_polsit,
                      vh_speed_o_weight,
                      vh_lnvalue_o_drv_age_x_pu,
                      flag_young_male.1,
                      vh_type.Tourism,
                      drv_drv2.Yes,
                      vh_lnvaluedep_o_vh_age,
                      town_surface_area,
                      vh_lnvalue_o_drv_age_x_pcppu,
                      pol_duration_o_age_x_pc,
                      AD_drv_vh_valuedep,
                      vh_fuel.Hybrid,
                      vh_weight,
                      pol_ncd_inc_risk_c,
                      vh_speed_o_town_x_pc,
                      vh_mm_x_lnspeed,
                      flag_veryyoung_male.1,
                      drv_sex1.M,
                      population,
                      vh_lnvalue_o_drv_age_x_pcpu,
                      vh_mm_o_drv_nyl_x_pc,
                      vh_mm_x_lnvaluedep_x_pu,
                      vh_lnvalue_o_drv_age_x_pc,
                      pol_pay_freq.Monthly,
                      vh_valuedep_o_town,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_speed,
                      vh_lnspeed_x_lnvaluedep_o_lnweight,
                      vh_lnvalue_o_vh_age,
                      AS_nyl_ncd,
                      vh_lnvalue_o_drv_age_x_pc,
                      AD_drv_ncd,
                      AS_nyl_age,
                      flag_senior.1,
                      pol_coverage.Med1,
                      pol_ncd_dec_risk_c,
                      vh_lnvalue_x_pu,
                      age_gap_gr.gap_low_17,
                      vh_lnspeed_x_lnvalue,
                      vh_age,
                      vh_lnvaluedep_x_lnspeed_o_vh_age,
                      vh_lnvalue_x_lnspeed_o_vh_age,
                      pol_usage.WorkPrivate,
                      age_gap_gr2.senior_as_occ,
                      vh_mm_x_lnvaluedep,
                      vh_age_p_drv_nyl_x_ncd_x_pu,
                      vh_valuedep,
                      drv_lic2_o_age,
                      pol_ncd_inc_risk_ca,
                      age_gap_gr2.kid_as_occ,
                      pol_ncd_dec_risk_ca,
                      AD_nyl_pol,
                      vh_value_o_town,
                      vh_lnvaluedep_o_vh_age_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcpu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcppu,
                      vh_mm_x_lnvaluedep_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pcppu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcpu,
                      pcppu_o_vh_age,
                      vh_lnvaluedep_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcppu,
                      AS_nyl_vh_age,
                      vh_lnvalue_o_vh_age_x_pc,
                      lnage_p_ncd_x_lnage_x_pu,
                      vh_lnvaluedep_o_vh_age_x_pc,
                      vh_mm_x_lnvalue_x_pcppu,
                      vh_lnvalue_x_pc,
                      vh_lnvaluedep_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      AS_drv_polsit,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcppu,
                      vh_mmord_x_lnvaluedep,
                      vh_mmpop_x_mmord,
                      vh_mmord_x_lnvalue,
                      vh_lnvaluedep_d_mmpop,
                      starts_with("PC",ignore.case = FALSE ))
  return(td)
}


trim_variables_broad <- function(td){
  td <- td %>% select(vh_mm_x_lnvaluedep_x_pcpu,
                      vh_value_o_town_x_pcpu,
                      AS_drv_vh_valuedep,
                      ncd_x_age_p_nyl1_x_pc,
                      pol_pay_freq.Quarterly,
                      pol_duration_o_nyl_x_pcpu,
                      vh_fuel.Gasoline,
                      drv_nyl2_o_age,
                      vh_lnspeed_x_lnvalue_o_lnweight,
                      AS_drv_vh_age,
                      pop_density_x_pcpu,
                      pol_pay_freq.Yearly,
                      pol_sit_o_duration,
                      vh_speed_o_density_x_pcpu,
                      pol_ncd_occ_c,
                      pol_ncd_dec,
                      pol_payd.Yes,
                      AD_drv_polsit,
                      vh_speed_o_weight,
                      vh_lnvalue_o_drv_age_x_pu,
                      flag_young_male.1,
                      vh_type.Tourism,
                      drv_drv2.Yes,
                      vh_lnvaluedep_o_vh_age,
                      town_surface_area,
                      vh_lnvalue_o_drv_age_x_pcppu,
                      pol_duration_o_age_x_pc,
                      AD_drv_vh_valuedep,
                      vh_fuel.Hybrid,
                      vh_weight,
                      pol_ncd_inc_risk_c,
                      vh_speed_o_town_x_pc,
                      vh_mm_x_lnspeed,
                      flag_veryyoung_male.1,
                      drv_sex1.M,
                      population,
                      vh_lnvalue_o_drv_age_x_pcpu,
                      vh_mm_o_drv_nyl_x_pc,
                      vh_mm_x_lnvaluedep_x_pu,
                      vh_lnvalue_o_drv_age_x_pc,
                      pol_pay_freq.Monthly,
                      vh_valuedep_o_town,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_speed,
                      vh_lnspeed_x_lnvaluedep_o_lnweight,
                      vh_lnvalue_o_vh_age,
                      AS_nyl_ncd,
                      vh_lnvalue_o_drv_age_x_pc,
                      AD_drv_ncd,
                      AS_nyl_age,
                      flag_senior.1,
                      pol_coverage.Med1,
                      pol_ncd_dec_risk_c,
                      vh_lnvalue_x_pu,
                      age_gap_gr.gap_low_17,
                      vh_lnspeed_x_lnvalue,
                      vh_age,
                      vh_lnvaluedep_x_lnspeed_o_vh_age,
                      vh_lnvalue_x_lnspeed_o_vh_age,
                      pol_usage.WorkPrivate,
                      age_gap_gr2.senior_as_occ,
                      vh_mm_x_lnvaluedep,
                      vh_age_p_drv_nyl_x_ncd_x_pu,
                      vh_valuedep,
                      drv_lic2_o_age,
                      pol_ncd_inc_risk_ca,
                      age_gap_gr2.kid_as_occ,
                      pol_ncd_dec_risk_ca,
                      AD_nyl_pol,
                      vh_value_o_town,
                      vh_lnvaluedep_o_vh_age_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcpu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcppu,
                      vh_mm_x_lnvaluedep_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pcppu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcpu,
                      pcppu_o_vh_age,
                      vh_lnvaluedep_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcppu,
                      AS_nyl_vh_age,
                      vh_lnvalue_o_vh_age_x_pc,
                      lnage_p_ncd_x_lnage_x_pu,
                      vh_lnvaluedep_o_vh_age_x_pc,
                      vh_mm_x_lnvalue_x_pcppu,
                      vh_lnvalue_x_pc,
                      vh_lnvaluedep_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      AS_drv_polsit,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcppu,
                      vh_mmord_x_lnvaluedep,
                      vh_mmpop_x_mmord,
                      vh_mmord_x_lnvalue,
                      vh_lnvaluedep_d_mmpop,
                      pol_no_claims_discount,
                      pol_coverage.Max,
                      pol_coverage.Med1,
                      pol_coverage.Med2,
                      pol_coverage.Min,
                      pol_duration,
                      pol_sit_duration,
                      pol_pay_freq.Biannual,
                      pol_pay_freq.Monthly,
                      pol_pay_freq.Quarterly,
                      pol_pay_freq.Yearly,
                      pol_payd.No,
                      pol_payd.Yes,
                      pol_usage.AllTrips,
                      pol_usage.Professional,
                      pol_usage.Retired,
                      pol_usage.WorkPrivate,
                      drv_sex1.F,
                      drv_sex1.M,
                      drv_age1,
                      drv_drv2.Yes,
                      drv_age2,
                      vh_age,
                      vh_fuel.Diesel,
                      vh_fuel.Gasoline,
                      vh_fuel.Hybrid,
                      vh_type.Commercial,
                      vh_type.Tourism,
                      vh_speed,
                      vh_value,
                      vh_weight,
                      population,
                      town_surface_area,
                      F_pol_pay_freq_change,
                      pol_ncd_occ,
                      pol_ncd_noocc,
                      pol_ncd_occ_c,
                      pol_ncd_noocc_c,
                      pol_ncd_inc,
                      pol_ncd_dec,
                      pol_ncd_inc_risk,
                      pol_ncd_dec_risk,
                      pol_ncd_inc_risk_c,
                      pol_ncd_dec_risk_c,
                      pol_ncd_inc_risk_ca,
                      pol_ncd_dec_risk_ca,
                      pc,
                      pu,
                      vf,
                      pcpu,
                      pcppu,
                      pol_sit_o_duration,
                      pol_duration_m_sit,
                      drv_nyl1,
                      drv_nyl2,
                      drv_lic1,
                      drv_lic2,
                      drv_age_avg,
                      drv_lic_avg,
                      drv_nyl_avg,
                      drv_nyl1_o_age,
                      drv_nyl2_o_age,
                      drv_lic1_o_age,
                      drv_lic2_o_age,
                      drv_age_p_nyl1,
                      drv_age_p_nyl2,
                      drv_age_p_lic1,
                      drv_age_p_lic2,
                      flag_young_male.1,
                      flag_veryyoung_male.1,
                      flag_senior.1,
                      age_gap_gr.gap_high_17,
                      age_gap_gr.gap_low_17,
                      age_gap_gr2.couple,
                      age_gap_gr2.kid_as_occ,
                      age_gap_gr2.senior_as_occ,
                      vh_valuedep,
                      pop_density,
                      starts_with("PC",ignore.case = FALSE ))
  return(td)
}

trim_variables_broad_leak <- function(td){
  td <- td %>% select(vh_mm_x_lnvaluedep_x_pcpu,
                      vh_value_o_town_x_pcpu,
                      AS_drv_vh_valuedep,
                      ncd_x_age_p_nyl1_x_pc,
                      pol_pay_freq.Quarterly,
                      pol_duration_o_nyl_x_pcpu,
                      vh_fuel.Gasoline,
                      drv_nyl2_o_age,
                      vh_lnspeed_x_lnvalue_o_lnweight,
                      AS_drv_vh_age,
                      pop_density_x_pcpu,
                      pol_pay_freq.Yearly,
                      pol_sit_o_duration,
                      vh_speed_o_density_x_pcpu,
                      pol_ncd_occ_c,
                      pol_ncd_dec,
                      pol_payd.Yes,
                      AD_drv_polsit,
                      vh_speed_o_weight,
                      vh_lnvalue_o_drv_age_x_pu,
                      flag_young_male.1,
                      vh_type.Tourism,
                      drv_drv2.Yes,
                      vh_lnvaluedep_o_vh_age,
                      town_surface_area,
                      vh_lnvalue_o_drv_age_x_pcppu,
                      pol_duration_o_age_x_pc,
                      AD_drv_vh_valuedep,
                      vh_fuel.Hybrid,
                      vh_weight,
                      pol_ncd_inc_risk_c,
                      vh_speed_o_town_x_pc,
                      vh_mm_x_lnspeed,
                      flag_veryyoung_male.1,
                      drv_sex1.M,
                      population,
                      vh_lnvalue_o_drv_age_x_pcpu,
                      vh_mm_o_drv_nyl_x_pc,
                      vh_mm_x_lnvaluedep_x_pu,
                      vh_lnvalue_o_drv_age_x_pc,
                      pol_pay_freq.Monthly,
                      vh_valuedep_o_town,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_speed,
                      vh_lnspeed_x_lnvaluedep_o_lnweight,
                      vh_lnvalue_o_vh_age,
                      AS_nyl_ncd,
                      vh_lnvalue_o_drv_age_x_pc,
                      AD_drv_ncd,
                      AS_nyl_age,
                      flag_senior.1,
                      pol_coverage.Med1,
                      pol_ncd_dec_risk_c,
                      vh_lnvalue_x_pu,
                      age_gap_gr.gap_low_17,
                      vh_lnspeed_x_lnvalue,
                      vh_age,
                      vh_lnvaluedep_x_lnspeed_o_vh_age,
                      vh_lnvalue_x_lnspeed_o_vh_age,
                      pol_usage.WorkPrivate,
                      age_gap_gr2.senior_as_occ,
                      vh_mm_x_lnvaluedep,
                      vh_age_p_drv_nyl_x_ncd_x_pu,
                      vh_valuedep,
                      drv_lic2_o_age,
                      pol_ncd_inc_risk_ca,
                      age_gap_gr2.kid_as_occ,
                      pol_ncd_dec_risk_ca,
                      AD_nyl_pol,
                      vh_value_o_town,
                      vh_lnvaluedep_o_vh_age_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pc,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcpu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcppu,
                      vh_mm_x_lnvaluedep_x_pcpu,
                      vh_mm_x_lnvaluedep_x_pcppu,
                      vh_lnvaluedep_x_lnspeed_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      vh_lnspeed_x_lnvaluedep_x_lnweight_x_pcpu,
                      pcppu_o_vh_age,
                      vh_lnvaluedep_x_pcpu,
                      lnage_p_ncd_x_lnage_x_pcpu,
                      vh_lnvalue_o_vh_age_x_pcppu,
                      AS_nyl_vh_age,
                      vh_lnvalue_o_vh_age_x_pc,
                      lnage_p_ncd_x_lnage_x_pu,
                      vh_lnvaluedep_o_vh_age_x_pc,
                      vh_mm_x_lnvalue_x_pcppu,
                      vh_lnvalue_x_pc,
                      vh_lnvaluedep_o_vh_age_x_pcppu,
                      vh_lnvaluedep_x_pc,
                      AS_drv_polsit,
                      vh_lnvalue_x_lnspeed_o_vh_age_x_pcppu,
                      vh_mmord_x_lnvaluedep,
                      vh_mmpop_x_mmord,
                      vh_mmord_x_lnvalue,
                      vh_lnvaluedep_d_mmpop,
                      pol_no_claims_discount,
                      pol_coverage.Max,
                      pol_coverage.Med1,
                      pol_coverage.Med2,
                      pol_coverage.Min,
                      pol_duration,
                      pol_sit_duration,
                      pol_pay_freq.Biannual,
                      pol_pay_freq.Monthly,
                      pol_pay_freq.Quarterly,
                      pol_pay_freq.Yearly,
                      pol_payd.No,
                      pol_payd.Yes,
                      pol_usage.AllTrips,
                      pol_usage.Professional,
                      pol_usage.Retired,
                      pol_usage.WorkPrivate,
                      drv_sex1.F,
                      drv_sex1.M,
                      drv_age1,
                      drv_drv2.Yes,
                      drv_age2,
                      vh_age,
                      vh_fuel.Diesel,
                      vh_fuel.Gasoline,
                      vh_fuel.Hybrid,
                      vh_type.Commercial,
                      vh_type.Tourism,
                      vh_speed,
                      vh_value,
                      vh_weight,
                      population,
                      town_surface_area,
                      F_pol_pay_freq_change,
                      pol_ncd_occ,
                      pol_ncd_noocc,
                      pol_ncd_occ_c,
                      pol_ncd_noocc_c,
                      pol_ncd_inc,
                      pol_ncd_dec,
                      pol_ncd_inc_risk,
                      pol_ncd_dec_risk,
                      pol_ncd_inc_risk_c,
                      pol_ncd_dec_risk_c,
                      pol_ncd_inc_risk_ca,
                      pol_ncd_dec_risk_ca,
                      pc,
                      pu,
                      vf,
                      pcpu,
                      pcppu,
                      pol_sit_o_duration,
                      pol_duration_m_sit,
                      drv_nyl1,
                      drv_nyl2,
                      drv_lic1,
                      drv_lic2,
                      drv_age_avg,
                      drv_lic_avg,
                      drv_nyl_avg,
                      drv_nyl1_o_age,
                      drv_nyl2_o_age,
                      drv_lic1_o_age,
                      drv_lic2_o_age,
                      drv_age_p_nyl1,
                      drv_age_p_nyl2,
                      drv_age_p_lic1,
                      drv_age_p_lic2,
                      flag_young_male.1,
                      flag_veryyoung_male.1,
                      flag_senior.1,
                      age_gap_gr.gap_high_17,
                      age_gap_gr.gap_low_17,
                      age_gap_gr2.couple,
                      age_gap_gr2.kid_as_occ,
                      age_gap_gr2.senior_as_occ,
                      vh_valuedep,
                      pop_density,
                      starts_with("PC",ignore.case = FALSE ))
  return(td)
}

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
#     $$\      $$\  $$$$$$\  $$$$$$$\  $$$$$$$$\ $$\       $$$$$$\  
#     $$$\    $$$ |$$  __$$\ $$  __$$\ $$  _____|$$ |     $$  __$$\ 
#     $$$$\  $$$$ |$$ /  $$ |$$ |  $$ |$$ |      $$ |     $$ /  \__|
#     $$\$$\$$ $$ |$$ |  $$ |$$ |  $$ |$$$$$\    $$ |     \$$$$$$\  
#     $$ \$$$  $$ |$$ |  $$ |$$ |  $$ |$$  __|   $$ |      \____$$\ 
#     $$ |\$  /$$ |$$ |  $$ |$$ |  $$ |$$ |      $$ |     $$\   $$ |
#     $$ | \_/ $$ | $$$$$$  |$$$$$$$  |$$$$$$$$\ $$$$$$$$\\$$$$$$  |
#     \__|     \__| \______/ \_______/ \________|\________|\______/
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# MODELS ####



fit_model <- function (x_raw, y_raw, 
                       p_split = 1, seed = 999, 
                       leakage_blocker = TRUE,
                       golden_feature = FALSE,
                       fit_stack  = TRUE,
                       fit_lgb    = TRUE,
                       fit_xgb    = TRUE,
                       fit_glmnet = TRUE,
                       fit_glm    = TRUE,
                       fit_gam    = FALSE,   #DON'T FIT GAM... I gave up on it.
                       fit_cat    = TRUE,
                       fit_rf     = TRUE,
                       fit_ens    = TRUE) {
  
  cat("*************************************************************** \n")
  cat("*******************                         ******************* \n")
  cat("*******************         WELCOME         ******************* \n")
  cat("*******************                         ******************* \n")
  cat("*******************         WARNING         ******************* \n")
  cat("*******************                         ******************* \n")
  cat("*************************************************************** \n")
  cat("***                                                         *** \n")
  cat("***  The following fitting procedure takes                  *** \n")
  cat("***  about 11 hours on a 10 core machine                    *** \n")
  cat("***                                                         *** \n")  
  cat("***    +- 2 hours for regular fit                           *** \n")
  cat("***    +- 9 hours for stacking fit                          *** \n")
  cat("***                                                         *** \n")
  cat("***  I live in a green energy province, but do you?  ;)     *** \n")
  cat("***                                                         *** \n")  
  cat("*************************************************************** \n")
  cat("*************************************************************** \n")
  cat("                                                                \n")
  cat("                                                                \n")
  cat("***  Week 9 update....                                          \n")
  cat("***  fit_model should be ran twice.                             \n")  
  cat("***  1) with golden_feature parameter = FALSE                   \n")
  cat("***  2) with golden_feature parameter = TRUE                    \n")  
  cat("***                                                             \n")
  cat("***  Then save both set of models,                              \n")
  cat("***  Using golden_feature parameter in the save_model()         \n")
  cat("                                                                \n")
  cat("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n")
  cat("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n")
  cat("XXX                                                         XXX \n")  
  cat("XXX   Additional Warning ...                                XXX \n")
  cat("XXX                                                         XXX \n")   
  cat("XXX   LightGBM could crash right away in the first steps    XXX \n")
  cat("XXX   It does for me, on Windows 10, RStudio portable       XXX \n")
  cat("XXX                                                         XXX \n")   
  cat("XXX   My bush fix:                                          XXX \n") 
  cat("XXX   1) Start a fresh session                              XXX \n")
  cat("XXX   2) (optional) Install/Reinstall lightgbm package      XXX \n")
  cat("XXX   3) ONLY load library(lightgbm).... NOTHING ELSE!      XXX \n") 
  cat("XXX   4) Run the first example from the basic walkthrough   XXX \n")
  cat("XXX   5) Should be good to go afterwards                    XXX \n")
  cat("XXX   6) If not, see step 1)                                XXX \n") 
  cat("XXX                                                         XXX \n") 
  cat("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n")
  cat("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n")
  cat("                                                                \n")
  
  # Basic Walkhthrough URL :
  # https://github.com/microsoft/LightGBM/blob/master/R-package/demo/basic_walkthrough.R

  cat("                                                                \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
  cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%     REGULAR TRAINING    %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("                                                                \n")
  print(Sys.time())
  cat("                                                                \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%    TRAINING :: START    %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("                                                                \n")
  
  
  
  
  # Initializing...
  trained_model <- vector("list",33)

  # Regular fit ....
  trained_model <- fit_model_regular(x_raw, y_raw,
                                     p_split, seed,
                                     leakage_blocker,
                                     golden_feature,
                                     fit_lgb    = fit_lgb,
                                     fit_xgb    = fit_xgb,
                                     fit_glmnet = fit_glmnet,
                                     fit_glm    = fit_glm,
                                     fit_gam    = fit_gam,
                                     fit_cat    = fit_cat,
                                     fit_rf     = fit_rf
                                     )
  
 
  cat("                                                                \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
  cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%     REGULAR TRAINING    %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("                                                                \n")
  print(Sys.time())
  cat("                                                                \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%    TRAINING :: DONE     %%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("                                                                \n") 
  
  cat("                                                                \n") 
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%    SUMMARY   REGULAR   %%%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%      on TRAIN SET      %%%%%%%%%%%%%%%%%%%% \n")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
  cat("                                                                \n")
  cat(" These are the predictions on the train data                    \n")
  cat("                                                                \n") 
  cat("                                                                \n")  
  print(summary(trained_model[[19]]))
  cat("                                                                \n")
  cat("                                                                \n")
  print(summary(trained_model[[20]]))
  cat("                                                                \n")
  cat("                                                                \n")
  print(summary(trained_model[[21]]))
  cat("                                                                \n")
  
  cat("                                                                \n")
  cat(" Using the just-fitted models back onto the same training data  \n")
  cat("   allows to calculate the offbalance factor required           \n")
  cat("      so that the total expected losses = total actual losses   \n")
  cat("                                                                \n")  
  #cat(" Computation of the adjustment will be done                     \n") 
  #cat("   in the predict expected claim                                \n")
 # cat("                                                                \n")  
  
  cat(" ?!? Calculating Off-Balance Factor on level 1                   \n")
  cat("                                                                 \n")
  claims_sum <- trained_model[[21]] %>% summarise(across(everything(), ~sum(.)))
  obf_train <- claims_sum %>% summarise(across(everything(), ~ y_pp/(.x) ))
  print(obf_train)
  
  
  # Need to save predictions from held out
  # If I'm in training mode...
  # This will be used down the road to verify 
  # if my ensemble/stack performs better than level 1 predictions on the test set
  
  if (p_split < 1) {
    
   
    cat("                                                                \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
    cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%     REGULAR PREDICT     %%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%        TEST SET         %%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
    cat("                                                                \n")    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #META DATA SPLIT ####
    set.seed(seed)
    inTrain  <- createDataPartition(y=y_raw$claim_amount, p=p_split, list=FALSE)
    x_raw_test <- x_raw[-inTrain,]
    y_raw_test <- y_raw[-inTrain,]
    y_raw_test <- as.data.frame(cbind(claim_amount = y_raw_test))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    #inflating the y, for meta training / validation
    x_raw_test             <- add_infl_factor(x_raw_test)
    y_raw_test$year        <- x_raw_test$year
    y_raw_test$infl_factor <- x_raw_test$infl_factor
    y_raw_test             <- preprocess_Y_data(y_raw_test)
    y_pp                   <- y_raw_test$claim_amount_inf_cap
    y_freq                 <- y_pp
    y_freq[y_freq>0]       <- 1    
    
    #removing infl factor and year from x
    x_raw_test <- x_raw_test %>% select(-infl_factor)
    
    #prediction the left-out fold
    xy_meta_test <- vector("list",3)
    pred_claims = predict_expected_claim_STACK(trained_model, 
                                                    x_raw_test, 
                                                    leakage_blocker=TRUE,
                                                    golden_feature=golden_feature,
                                                     fit_lgb    = fit_lgb,
                                                     fit_xgb    = fit_xgb,
                                                     fit_glmnet = fit_glmnet,
                                                     fit_glm    = fit_glm,
                                                     fit_gam    = fit_gam,
                                                     fit_cat    = fit_cat,
                                                     fit_rf     = fit_rf
                                               )
    
    
    pred_claims_freq  <- pred_claims[[1]]
    pred_claims_sev   <- pred_claims[[2]]
    pred_claims_lcost <- pred_claims[[3]]
    
    xy_meta_test[[1]]  = cbind(pred_claims_freq,  y_freq = y_freq)
    xy_meta_test[[2]]  = cbind(pred_claims_sev,   y_pp = y_pp)
    xy_meta_test[[3]]  = cbind(pred_claims_lcost, y_pp = y_pp)    

    #xy_meta_test = cbind(pred_claims_test, y_pp = y_pp)
    
    cat("                                                                \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%  STRUCTURE  META TEST  %%%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
    cat("                                                                \n")
    print(str(xy_meta_test[[1]]))
    cat("                                                                \n")
    cat("                                                                \n")
    print(str(xy_meta_test[[2]]))
    cat("                                                                \n")
    cat("                                                                \n")
    print(str(xy_meta_test[[3]]))
    cat("                                                                \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%   SUMMARY  META TEST   %%%%%%%%%%%%%%%%%%%% \n")
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
    cat("                                                                \n")
    print(summary(xy_meta_test[[1]]))
    cat("                                                                \n")
    cat("                                                                \n")
    print(summary(xy_meta_test[[2]]))
    cat("                                                                \n")
    cat("                                                                \n")
    print(summary(xy_meta_test[[3]]))
    cat("                                                                \n")
    
    
  }
  else if (p_split == 1) {
    cat("                                                                \n") 
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
    cat("%%%%%%%%%%%%%%%%%%%      SPLIT @ 100%       %%%%%%%%%%%%%%%%%%% \n")     
    cat("%%%%%%%%%%%%%%%%%%%  NO TEST SET AVAILABLE  %%%%%%%%%%%%%%%%%%% \n") 
    cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n") 
    cat("                                                                \n") 
    xy_meta_test <- 123
    
  }
  
  
  

  if (fit_stack == TRUE) {
  
  cat("                                                                                                                                     \n")    
  cat("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       \n")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       \n")
  cat("///////////////////////////////////////////     MOVING ON TO STACKING   //////////////////////////////////////////////////////       \n")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       \n")
  cat("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////       \n")
  cat("                                                                                                                                     \n")  
    
  cat("                             \n ")   
  cat("  Enjoy this hot coffee cup! \n ")
  cat("                             \n ")  
  cat("              )  (           \n" )
  cat("            (   ) )          \n" )
  cat("             ) ( (           \n" )
  cat("           _______)_         \n" )
  cat("        .-'---------|        \n" )
  cat("       ( C|   A I   |        \n" )
  cat("        '-.  CROWD  |        \n" )
  cat("          '_________'        \n" )
  cat("           '-------'         \n" )
  cat("                             \n ") 
  cat("                             \n ") 

  
  # Stacking steps ....
  stack_models <- fit_model_staked(x_raw, y_raw,
                                            p_split, seed,
                                            leakage_blocker,
                                            golden_feature,
                                             fit_lgb    = fit_lgb,
                                             fit_xgb    = fit_xgb,
                                             fit_glmnet = fit_glmnet,
                                             fit_glm    = fit_glm,
                                             fit_gam    = fit_gam,
                                             fit_cat    = fit_cat,
                                             fit_rf     = fit_rf,
                                             fit_ens    = fit_ens)
  
  cat("                                                                 \n")
  cat(" ?!? Calculating Off-Balance Factor on level 2                   \n")
  cat("                                                                 \n")
  claims_sum_stack <- stack_models[[1]][[4]] %>% summarise(across(everything(), ~sum(.)))
  obf_train_stack  <- claims_sum_stack %>% summarise(across(everything(), ~ y_pp/(.x) ))
  print(obf_train_stack)
  obf_train = cbind(obf_train,obf_train_stack)
  print(obf_train)
  
  
  # OUTPUT ####
  
  trained_model[[23]] <- stack_models[[1]]  #xy_meta_train
  trained_model[[24]] <- xy_meta_test       #xy_meta_test
  
  trained_model[[25]] <- stack_models[[2]]  #xgb_ens
  trained_model[[26]] <- stack_models[[3]]  #ann_ens
  trained_model[[27]] <- stack_models[[4]]  #rf_ens
  trained_model[[28]] <- stack_models[[5]]  #lgb_ens
  trained_model[[29]] <- obf_train          #obf factor


  }
  else if (fit_stack == FALSE) {
    trained_model[[24]] <- xy_meta_test 
  }
  
  return(trained_model)
}


fit_model_regular <- function (x_raw, y_raw, 
                               p_split = 1, seed = 999, 
                               leakage_blocker = FALSE,
                               golden_feature = FALSE,
                               fit_lgb    = TRUE,
                               fit_xgb    = TRUE,
                               fit_glmnet = TRUE,
                               fit_glm    = TRUE,
                               fit_gam    = FALSE,
                               fit_cat    = TRUE,
                               fit_rf     = TRUE) {
  

  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #META DATA SPLIT ####
  set.seed(seed)
  inTrain  <- createDataPartition(y=y_raw$claim_amount, p=p_split, list=FALSE)
  x_raw <- x_raw[inTrain,]
  y_raw <- y_raw[inTrain,]
  y_raw <- as.data.frame(cbind(claim_amount = y_raw))
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  
  # As we train the models, 
  # we will retain a prediction to do stacking at the end.
  cat("                                                              \n")
  cat("**** Preprocessing for LightGBM & xgboost **** \n")
  x_id_policy <- x_raw$id_policy

  #x_clean <- preprocess_X_data(x_raw, one_hot = FALSE, target_encoding = FALSE)
  x_clean <- preprocess_X_data(x_raw,
                               one_hot = TRUE,
                               target_encoding = TRUE,
                               leave_one_out = FALSE,
                               leakage_blocker = leakage_blocker,
                               golden_feature = golden_feature,
                               keep_vh_make_model = FALSE)
  
  x_clean_saved_for_rf <- x_clean

  y_raw$year <- x_clean$year
  y_raw$infl_factor <- x_clean$infl_factor
  
  y_clean <- preprocess_Y_data(y_raw)
  y_claim_inf <- y_clean$claim_amount_inf_cap
  y_claim_reg <- y_clean$claim_amount_cap
  y_claim_occ <- y_clean$claim_occ

  ####keep a vector of inflation factor for prediction and ensembling
  infl <- x_clean$infl_factor
  y_pp <- y_clean$claim_amount_inf_cap
  ####

  #Generating Features
  #Those features will only be useful for final test, when we will price Year 5 of contracts on known policies.
  #For RMSE leaderboard, we don't have access to actual historical claims


  xy_inf <- cbind(x_clean,y_claim_inf)

  xy_inf <- xy_inf[ , !(names(xy_inf)=="year")]
  xy_inf <- xy_inf[ , !(names(xy_inf)=="infl_factor")]

  xy_inf_occ <- cbind(xy_inf,y_claim_occ)
  #str(xy_inf_occ,list.len=ncol(xy_inf_occ))


  formula_all_reg = y_claim_reg ~ .
  formula_all_inf = y_claim_inf ~ .
  formula_all_occ = y_claim_occ ~ .
  formula_basic = y_claim_reg ~ vh_age + vh_speed + vh_weight + vh_value + drv_age_avg + population + town_surface_area + vh_make_model_enc


  #### **DATA SPLIT ############################################################

  #set.seed(seed)
  #str(xy_inf_occ, list.len=ncol(xy_inf_occ))
  #inTrain  <- createDataPartition(y=xy_inf_occ$y_claim_inf, p=p_split, list=FALSE)

  training_pp   <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_pp    <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]

  training_freq <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_inf") ]
  #testing_freq  <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_inf") ]

  training_sev  <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_sev   <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]

  training_sev  <- training_sev %>% filter(y_claim_inf > 0)
  #testing_sev   <- testing_sev  %>% filter(y_claim_inf > 0)


  y_train_pp     = training_pp$y_claim_inf
  #y_test_pp      = testing_pp$y_claim_inf

  y_train_freq   = training_freq$y_claim_occ
  #y_test_freq    = testing_freq$y_claim_occ

  y_train_freq_n = training_freq %>% mutate(y_claim_nb = ifelse(y_claim_occ=="Y",1,0)) %>% select(y_claim_nb) %>% pull()
  #y_test_freq_n  = testing_freq %>% mutate(y_claim_nb = ifelse(y_claim_occ=="Y",1,0)) %>% select(y_claim_nb)

  y_train_sev    = training_sev$y_claim_inf
  #y_test_sev     = testing_sev$y_claim_inf


  #cl <- makeCluster(detectCores())
  #registerDoParallel(cl)
  #cl

  
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 #### LIGHTGBM ##############################################################
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  
  if (fit_lgb == TRUE) {
    #### LGB - FREQ ####
    cat("                                                              \n")
    cat("++ Training LightGBM FREQ ++ \n")
    
    tic()
    x_train <- training_freq[,names(training_freq) != "y_claim_occ"]
    y_train = y_train_freq_n
    
    x_train_sparse = Matrix(as.matrix(x_train),sparse=TRUE)
    dtrain = lgb.Dataset(data=x_train_sparse, label=y_train)
    
    cat("+++ Cross-Validation to find best number of iterations +++    \n")
    bstCV <- lgb.cv(    data = dtrain
                        , num_leaves = 5 
                        , max_depth = 3
                        , min_data_in_leaf = 20
                        , min_sum_hessian_in_leaf = 20
                        , feature_fraction = 0.8
                        , bagging_fraction = 0.8
                        , bagging_freq = 1
                        , lambda_l1 = 0.4
                        , lambda_l2 = 0.2
                        , min_gain_to_split = 0.4
                        , learning_rate = 0.01
                        , num_iterations = 7000
                        , nthread = 8L
                        , eval = "binary_logloss"
                        , objective = "binary"
                        , early_stopping_round = 20
                        , eval_freq = 20
                        , nfold = 5
                        , stratified = TRUE
                        , verbose = -1
                        
    )
    #ok tuned 2021/02/10
    cat("+++ Best number of iteration is :", bstCV$best_iter , " \n")
    cat("                                                              \n")
    model_lgb_freq <- lgb.train(
                                data = dtrain
                                , num_leaves = 5
                                , max_depth = 3
                                , min_data_in_leaf = 20
                                , min_sum_hessian_in_leaf = 20
                                , feature_fraction = 0.8
                                , bagging_fraction = 0.8
                                , bagging_freq = 1
                                , lambda_l1 = 0.4
                                , lambda_l2 = 0.2
                                , min_gain_to_split = 0.4
                                , learning_rate = 0.01 
                                , num_iterations = bstCV$best_iter
                                , nthread = 8L
                                , eval = "binary_logloss"
                                , objective = "binary" 
                                , verbose = -1
                                
    )
    toc()
    
    
    expected_lgb_freq = predict(model_lgb_freq, x_train_sparse)
  
    
    
    
    #### LGB - SEV ####
    cat("                                                              \n")
    cat("++ Training LightGBM SEV ++ \n")
    
    tic()
    x_train <- training_sev[,names(training_sev) != "y_claim_inf"]
    y_train = y_train_sev
    
    x_train_sparse = Matrix(as.matrix(x_train),sparse=TRUE)
    dtrain = lgb.Dataset(data=x_train_sparse, label=y_train)
    
    cat("+++ Cross-Validation to find best number of iterations +++     \n")
    bstCV <- lgb.cv(    data = dtrain
                        , num_leaves = 5
                        , max_depth = 3
                        , min_data_in_leaf = 80
                        , min_sum_hessian_in_leaf = 120
                        , feature_fraction = 0.9
                        , bagging_fraction = 0.9
                        , bagging_freq = 1
                        , lambda_l1 = 0.4
                        , lambda_l2 = 0.3
                        , min_gain_to_split = 0.2
                        , learning_rate = 0.01
                        , num_iterations = 7000
                        , nthread = 8L
                        , eval = "gamma_deviance"
                        , objective = "gamma"
                        , early_stopping_round = 20
                        , eval_freq = 20
                        , nfold = 5
                        , stratified = TRUE
                        , verbose = -1
                        #, verbose_eval = FALSE
    )
    #ok tuned 2021/02/10
    cat("+++ Best number of iteration is :", bstCV$best_iter , " \n")
    cat("                                                              \n")
    model_lgb_sev <- lgb.train(
                                data = dtrain
                                , num_leaves = 5
                                , max_depth = 3
                                , min_data_in_leaf = 80
                                , min_sum_hessian_in_leaf = 120
                                , feature_fraction = 0.9
                                , bagging_fraction = 0.9
                                , bagging_freq = 1
                                , lambda_l1 = 0.4
                                , lambda_l2 = 0.3
                                , min_gain_to_split = 0.2
                                , learning_rate = 0.01 
                                , num_iterations = bstCV$best_iter
                                #, valids = valids
                                , nthread = 8L
                                , eval = "gamma_deviance"
                                , objective = "gamma" 
                                , verbose = -1
                                #, verbose_eval = FALSE
    )
    toc()
    
    
    #### LGB - LOSS COST ####
    cat("                                                              \n")
    cat("++ Training LightGBM LOSS COST ++ \n")
    
    tic()
    x_train <- training_pp[,names(training_pp) != "y_claim_inf"]
    y_train = training_pp$y_claim_inf
    
    x_train_sparse = Matrix(as.matrix(x_train),sparse=TRUE)
    dtrain = lgb.Dataset(data=x_train_sparse, label=y_train)
    
    cat("+++ Cross-Validation to find best number of iterations +++    \n")
    bstCV <- lgb.cv(    data = dtrain
                        , num_leaves = 31
                        , max_depth = 5
                        , min_data_in_leaf = 50
                        , min_sum_hessian_in_leaf = 10
                        , feature_fraction = 0.9
                        , bagging_fraction = 0.5
                        , bagging_freq = 3
                        , lambda_l1 = 0.4
                        , lambda_l2 = 0.4
                        , min_gain_to_split = 0.2
                        , learning_rate = 0.01
                        , num_iterations = 7000
                        , nthread = 8
                        , eval = c("rmse")
                        , objective = "tweedie"
                        , tweedie_variance_power = 1.4719
                        , early_stopping_round = 20
                        , eval_freq = 20
                        , nfold = 5
                        , stratified = TRUE
                        , verbose = -1
                        #, verbose_eval = FALSE
    )
    # old tune... 
    cat("+++ Best number of iteration is :", bstCV$best_iter , " \n")
    cat("                                                              \n")
    model_lgb_lcost <- lgb.train(
      data = dtrain
      , num_leaves = 31
      , max_depth = 5
      , min_data_in_leaf = 50
      , min_sum_hessian_in_leaf = 10
      , feature_fraction = 0.9
      , bagging_fraction = 0.5
      , bagging_freq = 3
      , lambda_l1 = 0.4
      , lambda_l2 = 0.4
      , min_gain_to_split = 0.2
      , learning_rate = 0.01 
      , num_iterations = bstCV$best_iter  #~330
      #, valids = valids
      , nthread = 8L
      , eval = "rmse"
      , objective = "tweedie"
      , tweedie_variance_power = 1.4719
      , verbose = -1
      #, verbose_eval = FALSE
    )
    
    
    toc()
    
    expected_lgb_lcost = predict(model_lgb_lcost, x_train_sparse)
    expected_lgb_sev = predict(model_lgb_sev, x_train_sparse)
    
    expected_lgb_sev[expected_lgb_sev<0] <- 0
    expected_lgb_sev[expected_lgb_sev>50000] <- 50000
    
    expected_lgb_fxs   = expected_lgb_freq * expected_lgb_sev / infl
    
    expected_lgb_lcost = expected_lgb_lcost / infl
    expected_lgb_lcost[expected_lgb_lcost<0] <- 0
    
  }
  else if (fit_lgb == FALSE){
    
    model_lgb_freq  <- 123 
    model_lgb_sev   <- 123  
    model_lgb_lcost <- 123
    expected_lgb_freq  = rep(0.10, length(infl))
    expected_lgb_sev   = rep(1000, length(infl)) 
    expected_lgb_fxs   = rep(101,  length(infl))
    expected_lgb_lcost = rep(100,  length(infl))
    
  }

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### XGB ###################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if (fit_xgb == TRUE) {
    #### XGB - FREQ ####
    cat("                                                              \n")
    cat("++ Training XGB FREQ ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5,
                                summaryFunction=twoClassSummary, 
                                classProbs=TRUE,
                                #savePredictions = TRUE, 
                                allowParallel=TRUE,
                                returnData = FALSE)
  
    xgb_grid_freq = expand.grid(nrounds =  seq(from =1000, to=1600, by=50),
                                  max_depth =5,
                                  eta = 0.005,
                                  gamma = 8.051563,
                                  colsample_bytree = 0.6749195,
                                  min_child_weight = 16.45606,
                                  subsample = 0.7508337 )
  
    model_xgb_freq <- caret::train(formula_all_occ, data=training_freq,
                                  method="xgbTree",
                                  trControl=fitControl,
                                  tuneGrid=xgb_grid_freq,
                                  objective ='binary:logistic',
                                  eval_metric = 'aucpr')  #for unbalanced, prefer to use aucpr instead of auc
    toc()
    
  
    #### XGB - SEV ####
    cat("                                                              \n")
    cat("++ Training XGB SEV ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5, allowParallel=TRUE, returnData = FALSE)
  
    xgb_grid_sev = expand.grid(nrounds = seq(from=1800, to=4000, by=50),
                                       max_depth = 2,
                                       eta = 0.005,
                                       gamma = 4.164731,
                                       colsample_bytree = 0.2655989,
                                       min_child_weight = 23.90476,
                                       subsample = 0.8447204)
  
    model_xgb_sev <- caret::train(formula_all_inf, data=training_sev,
                                 method="xgbTree",
                                 trControl=fitControl,
                                 tuneGrid=xgb_grid_sev,
                                 objective = "reg:gamma", #'reg:squarederror',
                                 eval_metric ="gamma-deviance")   #"rmse" )
    toc()
    
    
  
    #### XGB - LOSS COST ####
     cat("                                                              \n")
     cat("++ Training XGB LOSS COST ++ \n")
     tic()
     fitControl <- trainControl( method = "cv", number = 5, allowParallel=TRUE, returnData = FALSE)
  
     xgb_grid_lcost = expand.grid(nrounds = seq (from =2250, to=4000, by=250),
                                 max_depth = 2,
                                 eta = 0.005,
                                 gamma = 0.1039497,
                                 colsample_bytree = 0.299389,
                                 min_child_weight = 24.54995,
                                 subsample = 0.9689756)
  
     model_xgb_lcost <- caret::train(formula_all_inf, data=training_pp, method="xgbTree", trControl=fitControl,
                                 tuneGrid=xgb_grid_lcost, objective ='reg:tweedie', tweedie_variance_power = 1.471429,
                                 eval_metric = "tweedie-nloglik@1.4714", na.action=na.pass)
  
     toc()
  
     expected_xgb_freq  = predict(model_xgb_freq,  newdata = x_clean, type = "prob", na.action=na.pass)[,"Y"]
     expected_xgb_sev   = predict(model_xgb_sev,   newdata = x_clean, type = "raw", na.action=na.pass)
     expected_xgb_lcost = predict(model_xgb_lcost, newdata = x_clean, type = "raw", na.action=na.pass)
     
     expected_xgb_fxs = expected_xgb_freq * expected_xgb_sev / infl
     expected_xgb_lcost = expected_xgb_lcost / infl
     expected_xgb_fxs[expected_xgb_fxs<0] <- 0
     expected_xgb_lcost[expected_xgb_lcost<0] <- 0
     
    #registerDoSEQ()
  
  }
  else if (fit_xgb == FALSE){
      
    model_xgb_freq  <- 123
    model_xgb_sev   <- 123  
    model_xgb_lcost <- 123
    expected_xgb_freq  = rep(0.11, length(infl))
    expected_xgb_sev   = rep(1100, length(infl)) 
    expected_xgb_fxs   = rep(111,  length(infl))
    expected_xgb_lcost = rep(110,  length(infl))
      
  }
   

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### GLMNET ################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  cat("                                                              \n")
  cat("#### Preprocessing for GLMNET #### \n")
  x_id_policy <- x_raw$id_policy

  x_clean <- preprocess_X_data(x_raw,
                               one_hot = TRUE,
                               target_encoding = TRUE,
                               leave_one_out = TRUE,
                               leakage_blocker = leakage_blocker,
                               golden_feature = golden_feature,
                               keep_vh_make_model = FALSE
  )
  


  y_raw$year <- x_clean$year
  y_raw$infl_factor <- x_clean$infl_factor

  y_clean <- preprocess_Y_data(y_raw)
  y_claim_inf <- y_clean$claim_amount_inf_cap
  y_claim_reg <- y_clean$claim_amount_cap
  y_claim_occ <- y_clean$claim_occ

  xy_inf <- cbind(x_clean,y_claim_inf)

  xy_inf <- xy_inf[ , !(names(xy_inf)=="year")]
  xy_inf <- xy_inf[ , !(names(xy_inf)=="infl_factor")]

  xy_inf_occ <- cbind(xy_inf,y_claim_occ)

  formula_all_inf = y_claim_inf ~ .

  #### **DATA SPLIT ############################################################

  #set.seed(seed)
  #str(xy_inf_occ, list.len=ncol(xy_inf_occ))
  #inTrain  <- createDataPartition(y=xy_inf_occ$y_claim_inf, p=p_split, list=FALSE)

  training_pp   <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]

  y_train_pp     = training_pp$y_claim_inf

  if (fit_glmnet == TRUE) {  
    cat("                                                              \n")
    cat("++ Training GLMNET LOSS COST ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5,
                                allowParallel = TRUE, returnData = FALSE, trim=TRUE)
  
    glmnet_grid  <- expand.grid( alpha = seq(from = 0, to = 1, by = 0.1) ,
                                 lambda = seq(0.0001, 5, length=20))
  
    model_glmnet_lcost <- caret::train(formula_all_inf, data=training_pp,
                                method="glmnet",
                                preProcess=c("center","scale"),
                                trControl=fitControl,
                                tuneGrid=glmnet_grid,
                                verbose = FALSE )
    toc()
  
    expected_glmnet_lcost = predict(model_glmnet_lcost, newdata = x_clean, type = "raw")  
    expected_glmnet_lcost = expected_glmnet_lcost / infl
    expected_glmnet_lcost[expected_glmnet_lcost<0] <- 0
    expected_glmnet_lcost[expected_glmnet_lcost>50000] <- 50000
  
  }
  else if (fit_glmnet == FALSE){
  
    model_glmnet_lcost <- 123
    expected_glmnet_lcost = rep(120,  length(infl))
      
  }  
    
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### GLM ###################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  cat("                                                              \n")
  cat("#### Preprocessing for GLM #### \n")
  x_id_policy <- x_raw$id_policy
  
  #x_clean <- preprocess_X_data(x_raw, 
  #                             one_hot = TRUE, 
  #                             target_encoding = TRUE, 
  #                             leave_one_out = TRUE )

  #x_clean was processed correctly for GLMNET, so no need to reprocess it.
  
  y_raw$year <- x_clean$year
  y_raw$infl_factor <- x_clean$infl_factor
  
  y_clean <- preprocess_Y_data(y_raw)
  y_claim_inf <- y_clean$claim_amount_inf_cap
  y_claim_reg <- y_clean$claim_amount_cap
  y_claim_occ <- y_clean$claim_occ
  
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
    
  #Training center, scale, pca.  
  cat("+ Center and Scaling for GLM + \n")
  cs_model <- preProcess(x_clean, method = c("center","scale"))
  cs_mat <- predict(cs_model, newdata = x_clean)
  
  cat("+ PCA for GLM + \n")
  pca_model <- preProcess(cs_mat, method = c("pca"))          
  pca_mat <- predict(pca_model, newdata = cs_mat)
  x_clean <- cbind(cs_mat,pca_mat)
  
  x_clean <- trim_variables(x_clean)
 
  
  
  #Generating Features
  #Those features will only be useful for final test, when we will price Year 5 of contracts on known policies.
  #For RMSE leaderboard, we don't have access to actual historical claims
  #xy <- cbind(x_clean,y_clean)  
  #xy$id_policy = x_id_policy
  #features_xy <- generating_features(xy)
  #xy <- xy[ , !(names(xy)=="id_policy")]
  
  #xy_reg <- cbind(x_clean,features_xy,y_claim_reg)
  #xy_inf <- cbind(x_clean,features_xy,y_claim_inf)
  #xy_occ <- cbind(x_clean,features_xy,y_claim_occ)  
  
  xy_inf <- cbind(x_clean,y_claim_inf)
  
  xy_inf_occ <- cbind(xy_inf,y_claim_occ)
  #str(xy_inf_occ,list.len=ncol(xy_inf_occ))
  
  #glmModel_r <- glm(formula_all, family=tweedie(var.power=1.4, link.power=0), data=xy)
  #glmModel_r
  #varImp(glmModel_r)
  
  formula_all_reg = y_claim_reg ~ .
  formula_all_inf = y_claim_inf ~ .
  formula_all_occ = y_claim_occ ~ .

  
  #### **DATA SPLIT ############################################################
  
  #set.seed(seed)
  #str(xy_inf_occ, list.len=ncol(xy_inf_occ))
  #inTrain  <- createDataPartition(y=xy_inf_occ$y_claim_inf, p=p_split, list=FALSE)
  
  training_pp   <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_pp    <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_freq <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_inf") ]
  #testing_freq  <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_inf") ]
  
  training_sev  <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_sev   <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_sev  <- training_sev %>% filter(y_claim_inf > 0)
  #testing_sev   <- testing_sev  %>% filter(y_claim_inf > 0)
  
  
  y_train_pp     = training_pp$y_claim_inf
  #y_test_pp      = testing_pp$y_claim_inf
  
  y_train_freq   = training_freq$y_claim_occ
  #y_test_freq    = testing_freq$y_claim_occ
  
  y_train_freq_n = training_freq %>% mutate(y_claim_nb = ifelse(y_claim_occ=="Y",1,0)) %>% select(y_claim_nb)
  #y_test_freq_n  = testing_freq %>% mutate(y_claim_nb = ifelse(y_claim_occ=="Y",1,0)) %>% select(y_claim_nb)
  
  y_train_sev    = training_sev$y_claim_inf
  #y_test_sev     = testing_sev$y_claim_inf
  
  if (fit_glm == TRUE) {  
    #### GLM - FREQ ####
    cat("                                                              \n")
    cat("++ Training GLM FREQ ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE,
                                classProbs=TRUE, 
                                savePredictions = TRUE, returnData = FALSE, trim=TRUE)
  
    model_glm_freq <- train(formula_all_occ, data=training_freq, method="glm", 
                                    family="binomial",  trControl=fitControl)
    toc()
    
    
    #### GLM - SEV ####
    cat("                                                              \n")
    cat("++ Training GLM SEV ++ \n") 
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE, returnData = FALSE, trim=TRUE)
    
    model_glm_sev <- train(formula_all_inf, data=training_sev, method="glm", 
                                   family=Gamma("log"), trControl=fitControl)
    toc()
    
    #### GLM - LOSS COST ####
    cat("                                                              \n")
    cat("++ Training GLM LOSS COST  ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE, returnData = FALSE, trim=TRUE)
  
    model_glm_lcost <- train(formula_all_inf, data=training_pp, method="glm", 
                                     family=tweedie(var.power=1.471429, link.power=0),
                                     trControl=fitControl)
    toc()
  
    expected_glm_freq  = predict(model_glm_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_glm_sev   = predict(model_glm_sev,   newdata = x_clean, type = "raw" )
    expected_glm_lcost = predict(model_glm_lcost, newdata = x_clean, type = "raw")  
    
    expected_glm_sev[expected_glm_sev>50000] <- 50000
    
    expected_glm_fxs   = expected_glm_freq * expected_glm_sev / infl
    expected_glm_lcost = expected_glm_lcost / infl
    
    expected_glm_fxs[expected_glm_fxs<0] <- 0
    expected_glm_lcost[expected_glm_lcost<0] <- 0
    expected_glm_lcost[expected_glm_lcost>50000] <- 50000
  }
  else if (fit_glm == FALSE){
      
      model_glm_freq  <- 123 
      model_glm_sev   <- 123  
      model_glm_lcost <- 123
      expected_glm_freq  = rep(0.13, length(infl))
      expected_glm_sev   = rep(1300, length(infl)) 
      expected_glm_fxs   = rep(131,  length(infl))
      expected_glm_lcost = rep(130,  length(infl))
      
  }
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### GAM ###################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  
  if (fit_gam == TRUE) {  
    #### GAM - FREQ ####
    cat("                                                              \n")
    cat("++ Training GAM FREQ ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE,
                                classProbs=TRUE, 
                                savePredictions = TRUE, returnData = FALSE, trim=TRUE)
    
    model_gam_freq <- train(formula_all_occ, data=training_freq, method="gam", 
                            family="binomial",  trControl=fitControl)
    toc()
    
    
    #### GAM - SEV ####
    cat("                                                              \n")
    cat("++ Training GAM SEV ++ \n") 
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE, returnData = FALSE, trim=TRUE)
    
    model_gam_sev <- train(formula_all_inf, data=training_sev, method="gam", 
                           family=Gamma("log"), trControl=fitControl)
    toc()
    
    #### GAM - LOSS COST ####
    cat("                                                              \n")
    cat("++ Training GAM LOSS COST  ++ \n")
    tic()
    fitControl <- trainControl( method = "cv", number = 5, 
                                allowParallel = TRUE, returnData = FALSE, trim=TRUE)
    
    model_gam_lcost <- train(formula_all_inf, data=training_pp, method="gam", 
                             family=tweedie(var.power=1.471429, link.power=0),
                             trControl=fitControl)
    toc()
    
    expected_gam_freq  = predict(model_gam_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_gam_sev   = predict(model_gam_sev,   newdata = x_clean, type = "raw" )
    expected_gam_lcost = predict(model_gam_lcost, newdata = x_clean, type = "raw")  
    
    expected_gam_sev[expected_gam_sev>50000] <- 50000
    
    expected_gam_fxs   = expected_gam_freq * expected_gam_sev / infl
    expected_gam_lcost = expected_gam_lcost / infl
    
    expected_gam_fxs[expected_gam_fxs<0] <- 0
    expected_gam_lcost[expected_gam_lcost<0] <- 0
    expected_gam_lcost[expected_gam_lcost>50000] <- 50000
  }
  else if (fit_gam == FALSE){
    
    model_gam_freq  <- 123 
    model_gam_sev   <- 123  
    model_gam_lcost <- 123
    expected_gam_freq  = rep(0.133, length(infl))
    expected_gam_sev   = rep(1330, length(infl)) 
    expected_gam_fxs   = rep(133,  length(infl))
    expected_gam_lcost = rep(132,  length(infl))
    
  }
  
    
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### CATBOOST ##############################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  cat("                                                              \n")
  cat("#### Preprocessing for CATBOOST #### \n")
  
  x_clean <- preprocess_X_data(x_raw, 
                               one_hot = FALSE, 
                               target_encoding = FALSE,
                               leakage_blocker = leakage_blocker,
                               golden_feature = golden_feature,
                               keep_vh_make_model = TRUE
  ) #for catboost, we don't one_hot encode
  
  y_raw$year <- x_clean$year
  y_raw$infl_factor <- x_clean$infl_factor
  
  y_clean <- preprocess_Y_data(y_raw)
  y_claim_inf <- y_clean$claim_amount_inf_cap
  y_claim_reg <- y_clean$claim_amount_cap
  y_claim_occ <- y_clean$claim_occ
  
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  xy_inf <- cbind(x_clean,y_claim_inf)
  
  xy_inf_occ <- cbind(xy_inf,y_claim_occ)
  
  
  #### **DATA SPLIT ############################################################
  
  #set.seed(seed)
  #inTrain  <- createDataPartition(y=xy_inf_occ$y_claim_inf, p=p_split, list=FALSE)
  
  training_pp   <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_pp    <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_freq <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_inf") ]
  #testing_freq  <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_inf") ]
  
  training_sev  <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_sev   <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_sev  <- training_sev %>% filter(y_claim_inf > 0)
  
  
  x_train_pp <- training_pp[,names(training_pp) != "y_claim_inf"]
  y_train_pp =  training_pp$y_claim_inf
  
  x_train_freq <- training_freq[,names(training_freq) != "y_claim_occ"]
  y_train_freq =  training_freq$y_claim_occ
  
  x_train_sev <- training_sev[,names(training_sev) != "y_claim_inf"]
  y_train_sev = training_sev$y_claim_inf
  
  if (fit_cat == TRUE){ 
    
    #### CAT - LOSS COST ####
    cat("                                                              \n")
    cat("++ Training CATBOOST LOSS COST  ++ \n")
    
    trainpool <- catboost.load_pool(data = x_train_pp, label = y_train_pp)
    
    fitParams     <- list(loss_function = 'Tweedie:variance_power=1.47143', #  'RMSE',  #"Logloss"
                          eval_metric='RMSE',
                          iterations = 655, 
                          border_count = 204,
                          depth = 8,
                          learning_rate = 0.01,  
                          l2_leaf_reg = 0.3633494,
                          rsm = 0.6037537,
                          metric_period=10,
                          thread_count = 8,
                          verbose = 0
                          #od_type="Iter",
                          #od_wait=40
    )
    
    model_cat_lcost  <- catboost.train(trainpool,
                                       params= fitParams
    )
   
    
    #### CAT - FREQ ####
    cat("                                                              \n")
    cat("++ Training CATBOOST FREQ  ++ \n")
    
    trainpool <- catboost.load_pool(data = x_train_freq, label = as.integer(y_train_freq))
    
    fitParams     <- list(loss_function = 'Logloss', 
                          #eval_metric='RMSE',
                          iterations =  1010, 
                          border_count = 72,
                          depth = 8,
                          learning_rate = 0.01,  
                          l2_leaf_reg = 0.4227727,
                          rsm = 0.6133443,
                          metric_period=1,
                          thread_count = 8,
                          verbose = 0
                          #od_type="Iter",
                          #od_wait=40
    )
    
    model_cat_freq  <- catboost.train(trainpool,
                                       params= fitParams
    )
    
    
    #### CAT - SEV ####
    cat("                                                              \n")
    cat("++ Training CATBOOST SEV  ++ \n")
    
    trainpool <- catboost.load_pool(data = x_train_sev, label = y_train_sev)
    
    fitParams     <- list(loss_function = 'RMSE', 
                          eval_metric='RMSE',
                          iterations =  249, 
                          border_count = 16,
                          depth = 2,
                          learning_rate = 0.01,  
                          l2_leaf_reg = 0.1897699,
                          rsm = 0.9508522,
                          metric_period=1,
                          thread_count = 8,
                          verbose = 0
                          #od_type="Iter",
                          #od_wait=40
    )
    
    model_cat_sev  <- catboost.train(trainpool,
                                      params= fitParams
    )
    
    trainpool <- catboost.load_pool(data = x_train_pp)
    
    expected_cat_freq = catboost.predict(model_cat_freq, trainpool,  prediction_type='Probability')   
    expected_cat_sev = catboost.predict(model_cat_sev, trainpool) 
    expected_cat_lcost = catboost.predict(model_cat_lcost, trainpool,  prediction_type='Exponent')   
    expected_cat_fxs   = expected_cat_freq * expected_cat_sev / infl
    expected_cat_lcost = expected_cat_lcost / infl
  
    expected_cat_fxs[expected_cat_fxs<0] <- 0
    expected_cat_lcost[expected_cat_lcost<0] <- 0
  }
  else if (fit_cat == FALSE){
      
    model_cat_freq  <- 123 
    model_cat_sev   <- 123  
    model_cat_lcost <- 123
    expected_cat_freq  = rep(0.14, length(infl))
    expected_cat_sev   = rep(1400, length(infl)) 
    expected_cat_fxs   = rep(141,  length(infl))
    expected_cat_lcost = rep(140,  length(infl))
      
  }  
  
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### RANDOM FOREST #########################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  cat("                                                              \n")
  cat("#### Preprocessing for RANDOM FOREST #### \n")
  
  #x_clean <- preprocess_X_data(x_raw, 
  #                             one_hot = TRUE, 
  #                             target_encoding = TRUE
  #) 
  
  x_clean <- x_clean_saved_for_rf #starting back from xgb processed
  
  if (golden_feature == TRUE) {
    x_gf <- x_clean %>% select(F_claim_cnt, 
                               F_claim_cnt_c, 
                               F_claim_amt_c,
                               F_claim_amt_avg,
                               F_cov)
  }
  
  y_raw$year <- x_clean$year
  y_raw$infl_factor <- x_clean$infl_factor
  
  y_clean <- preprocess_Y_data(y_raw)
  y_claim_inf <- y_clean$claim_amount_inf_cap
  y_claim_reg <- y_clean$claim_amount_cap
  y_claim_occ <- y_clean$claim_occ
  
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  

  if (leakage_blocker == TRUE) {

    x_trim <- trim_variables_broad_leak(x_clean) 
    
  }
  else if (leakage_blocker == FALSE)  {
    x_trim <- trim_variables_broad(x_clean)
  }
  
  
  if (golden_feature == TRUE) {
    x_trim = cbind(x_trim,x_gf)
  }
  
  xy_inf <- cbind(x_trim,y_claim_inf)
  
  xy_inf_occ <- cbind(xy_inf,y_claim_occ)
  
  
  #### **DATA SPLIT ############################################################
  
  #set.seed(seed)
  #inTrain  <- createDataPartition(y=xy_inf_occ$y_claim_inf, p=p_split, list=FALSE)
  
  training_pp   <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_pp    <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_freq <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_inf") ]
  #testing_freq  <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_inf") ]
  
  training_sev  <- xy_inf_occ[,   !(names(xy_inf_occ)=="y_claim_occ") ]
  #testing_sev   <- xy_inf_occ[-inTrain,  !(names(xy_inf_occ)=="y_claim_occ") ]
  
  training_sev  <- training_sev %>% filter(y_claim_inf > 0)
  
  
  x_train_pp <- training_pp[,names(training_pp) != "y_claim_inf"]
  y_train_pp =  training_pp$y_claim_inf
  
  x_train_freq <- training_freq[,names(training_freq) != "y_claim_occ"]
  y_train_freq =  training_freq$y_claim_occ
  
  x_train_sev <- training_sev[,names(training_sev) != "y_claim_inf"]
  y_train_sev = training_sev$y_claim_inf
  
  
  if (fit_rf == TRUE) {  
    #### RF - FREQ ####
    cat("                                                              \n")
    cat("++ Training RANDOM FOREST FREQ  ++ \n")
    
    tic()
    model_rf_freq <- ranger(
                              formula         = formula_all_occ,
                              probability     = TRUE,  #important to have!
                              data            = training_freq, 
                              num.trees       = 1450,
                              mtry            = 1,
                              min.node.size   = 1,
                              replace         = TRUE,
                              sample.fraction = 0.95,
                              splitrule       = "hellinger"  ,
                              #verbose         = TRUE,
                              seed            = seed,
                              respect.unordered.factors = 'order',
                              importance      = 'impurity' ,
                              num.threads     = 8
    )
    toc()
    
    
    e = predict(model_rf_freq, data=x_trim, type="response")
    expected_rf_freq = e$predictions[,"Y"] 
    
  
    
    
    #### RF - SEV ####
    cat("                                                              \n")
    cat("++ Training RANDOM FOREST SEV  ++ \n")
    
    tic()
    model_rf_sev <- ranger(
                            formula         = formula_all_inf, 
                            data            = training_sev, 
                            num.trees       = 1450,
                            mtry            = 11,
                            min.node.size   = 120,
                            replace         = TRUE,
                            sample.fraction = 0.4,
                            splitrule       = "extratrees" ,
                            #verbose         = TRUE,
                            seed            = seed,
                            importance      = 'impurity' ,
                            num.threads     = 8
                          )
    toc()
    #ok, parameters tuned on 2021/02/10
    
    e = predict(model_rf_sev, data=x_trim, type="response")
    expected_rf_sev = e$predictions / infl
    
    expected_rf_sev[expected_rf_sev<0] <- 0
    expected_rf_sev[expected_rf_sev>50000] <- 50000  
    
    
    #### RF - LOSS COST ####
    cat("                                                              \n")
    cat("++ Training RANDOM FOREST LOSS COST  ++ \n")
    
    tic()
    model_rf_lcost <- ranger(
                          formula         = formula_all_inf, 
                          data            = training_pp, 
                          num.trees       = 1450,
                          mtry            = 13,
                          min.node.size   = 55,
                          replace         = TRUE,
                          sample          = 0.7,
                          splitrule       = "maxstat",
                          #old parameters
                          #mtry            = 1,
                          #min.node.size   = 35,
                          #replace         = FALSE,
                          #sample.fraction = 0.6,
                          #splitrule       = "??" ,
                          #verbose         = TRUE,
                          seed            = seed,
                          respect.unordered.factors = 'order',
                          importance      = 'impurity' ,
                          num.threads     = 8
    )
    toc()
    #ok, parameters tuned on 2021/02/10
    
    e = predict(model_rf_lcost, data=x_trim, type="response")
    
    expected_rf_fxs   = expected_rf_freq * expected_rf_sev / infl
    
    expected_rf_lcost = e$predictions / infl
    
    expected_rf_lcost[expected_rf_lcost<0] <- 0
    
  }
  else if (fit_rf == FALSE){
      
      model_rf_freq  <- 123 
      model_rf_sev   <- 123  
      model_rf_lcost <- 123
      expected_rf_freq  = rep(0.15, length(infl))
      expected_rf_sev   = rep(1500, length(infl)) 
      expected_rf_fxs   = rep(151,  length(infl))
      expected_rf_lcost = rep(150,  length(infl))
      
  }  
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### ENSEMBLING ############################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  ens_models_freq  <- data.frame(cbind( xgb_freq     = expected_xgb_freq,
                                        glm_freq     = expected_glm_freq,
                                        #gam_freq     = expected_gam_freq,
                                        cat_freq     = expected_cat_freq,
                                        rf_freq      = expected_rf_freq,
                                        lgb_freq     = expected_lgb_freq,
                                        y_pp         = y_pp) )  
  
  ens_models_sev   <- data.frame(cbind( xgb_sev     = expected_xgb_sev,
                                        glm_sev     = expected_glm_sev,
                                        #gam_sev     = expected_gam_sev,
                                        cat_sev     = expected_cat_sev,
                                        rf_sev      = expected_rf_sev,
                                        lgb_sev     = expected_lgb_sev,
                                        y_pp        = y_pp) )  
  
  
  
  ens_models_lcost <- data.frame(cbind( xgb_fxs      = expected_xgb_fxs,
                                        xgb_lcost    = expected_xgb_lcost,
                                        glmnet_lcost = expected_glmnet_lcost,
                                        glm_fxs      = expected_glm_fxs,
                                        glm_lcost    = expected_glm_lcost,
                                        #gam_fxs      = expected_gam_fxs,
                                        #gam_lcost    = expected_gam_lcost,
                                        cat_fxs      = expected_cat_fxs,
                                        cat_lcost    = expected_cat_lcost,
                                        rf_fxs       = expected_rf_fxs,
                                        rf_lcost     = expected_rf_lcost,
                                        lgb_fxs      = expected_lgb_fxs,
                                        lgb_lcost    = expected_lgb_lcost,
                                        y_pp         = y_pp) )
  
  
  # x_train_ens = ens_models[, -9]
  # y_train_ens = ens_models[,  9]
  # 
  # trainpool <- catboost.load_pool(data = x_train_ens, label = y_train_ens)
  # 
  # fitParams     <- list(loss_function = 'Tweedie:variance_power=1.47143',
  #                       eval_metric='RMSE',
  #                       iterations = 1975, 
  #                       border_count = 326,
  #                       depth = 9,
  #                       learning_rate = 0.005,  
  #                       l2_leaf_reg = 0.00001,
  #                       rsm = 0.64,
  #                       metric_period=10,
  #                       thread_count = 10,
  #                       verbose = 0
  #                       #od_type="Iter",
  #                       #od_wait=40
  # )
  # 
  # model_cat_ens_tw  <- catboost.train(trainpool,
  #                                     params= fitParams
  # )
  # 
  # 
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### OUTPUT ################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  # model_xgb_freq = "placeholder"
  # model_xgb_sev = "placeholder"
  # model_xgb_lcost = "placeholder"
  # model_glmnet_lcost = "placeholder"
  # model_glm_freq = "placeholder"
  # model_glm_sev = "placeholder"
  # model_glm_lcost = "placeholder"
  # model_cat_freq = "placeholder"
  # model_cat_sev = "placeholder" 
  # model_cat_lcost = "placeholder"
  # model_rf_lcost = "placeholder"
  # #model_lgb_lcost = "placeholder"
  # cs_model = "placeholder"
  # pca_model = "placeholder"
  # #ens_models = "placeholder"  
  
  
  
  trained_model <- list(    model_xgb_freq      # 1
                          , model_xgb_sev       # 2
                          , model_xgb_lcost     # 3
                          , model_glmnet_lcost  # 4 
                          , model_glm_freq      # 5 
                          , model_glm_sev       # 6 
                          , model_glm_lcost     # 7 
                          #, model_gam_freq      # 8
                          #, model_gam_sev       # 9
                          #, model_gam_lcost     # 10
                          , model_cat_freq      # 8
                          , model_cat_sev       # 9
                          , model_cat_lcost     # 10
                          , model_rf_freq       # 11
                          , model_rf_sev        # 12
                          , model_rf_lcost      # 13
                          , model_lgb_freq      # 14
                          , model_lgb_sev       # 15
                          , model_lgb_lcost     # 16
                          , cs_model            # 17
                          , pca_model           # 18
                          , ens_models_freq     # 19
                          , ens_models_sev      # 20
                          , ens_models_lcost    # 21 
                          , inTrain             # 22
                       )
  
  #stopCluster() 
  #registerDoSEQ()
  
   # I will want to have mutiple models down the road....
   
  #https://stackoverflow.com/questions/14954399/put-multiple-data-frames-into-list-smart-way
  #https://stackoverflow.com/questions/12268944/storing-multiple-data-frames-into-one-data-structure-r
   #https://stackoverflow.com/questions/59733326/merging-two-lm-objects-in-one-in-r
   
  return(trained_model)
}



fit_model_staked <- function(x_raw, y_raw,
                             p_split = 1, seed = 999,
                             leakage_blocker = TRUE,
                             golden_feature = FALSE,
                             fit_lgb    = TRUE,
                             fit_xgb    = TRUE,
                             fit_glmnet = TRUE,
                             fit_glm    = TRUE,
                             fit_gam    = FALSE,
                             fit_cat    = TRUE,
                             fit_rf     = TRUE,
                             fit_ens    = TRUE) {
  
  
  
  #Hold-out and remove it.
  set.seed(seed)
  inTrain_meta  <- createDataPartition(y=y_raw$claim_amount, p=p_split, list=FALSE)
  x_raw <- x_raw[inTrain_meta,]
  y_raw <- y_raw[inTrain_meta,]
  y_raw <- as.data.frame(cbind(claim_amount = y_raw))

  #Split into folds what is left of the train data.
  inFolds_stk  <- createFolds(y=y_raw$claim_amount, k = 5, list=TRUE, returnTrain = FALSE)
  
  stk_train_1 <- x_raw[inFolds_stk[[1]],]
  stk_train_2 <- x_raw[inFolds_stk[[2]],]
  stk_train_3 <- x_raw[inFolds_stk[[3]],]
  stk_train_4 <- x_raw[inFolds_stk[[4]],]
  stk_train_5 <- x_raw[inFolds_stk[[5]],]
  
  folds_stk_train <- list(stk_train_1, stk_train_2, stk_train_3, stk_train_4, stk_train_5)
  
  #initializing lists
  data_x_stk_train     <- vector("list",5)
  data_y_stk_train     <- vector("list",5)
  stk_train_pred_freq  <- vector("list",5)
  stk_train_pred_sev   <- vector("list",5)
  stk_train_pred_lcost <- vector("list",5)
  model_stk_train      <- vector("list",5)
  

  # We need to train on 4/5 and predict on last 5
  
  for (i in 1:5) {
    
    cat("                                                                              \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$      STACKING LOOP      $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$           ", i ,"           $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("                                                               \n")
    print(Sys.time())
    cat("                                                               \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$    TRAINING :: START    $$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("                                                               \n")
    
    #Removing the fold.
    x_stk_train <- x_raw[-inFolds_stk[[i]],] 
    y_stk_train <- as.data.frame(cbind(claim_amount = y_raw[-inFolds_stk[[i]],]))
    
    data_x_stk_train[[i]] = x_stk_train
    data_y_stk_train[[i]] = y_stk_train
    
    #Fitting model
    tempmodel = fit_model_regular(x_stk_train, y_stk_train, 
                          p_split=p_split, seed = seed, 
                          leakage_blocker = leakage_blocker,
                          golden_feature = golden_feature,
                          fit_lgb    = fit_lgb,
                          fit_xgb    = fit_xgb,
                          fit_glmnet = fit_glmnet,
                          fit_glm    = fit_glm,
                          fit_gam    = fit_gam,
                          fit_cat    = fit_cat,
                          fit_rf     = fit_rf
                          ) 
    
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$    TRAINING  :: DONE    $$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$   PREDICTING :: START   $$$$$$$$$$$$$$$$$$$ \n") 
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("                                                                \n")
    print(Sys.time())
    cat("                                                               \n")
    
    #Identifying the left-out fold
    x_stk_test <- x_raw[inFolds_stk[[i]],] 
    y_stk_test <- as.data.frame(cbind(claim_amount = y_raw[inFolds_stk[[i]],]))
    
    #inflating the y, for meta training
    x_stk_test             <- add_infl_factor(x_stk_test)
    y_stk_test$year        <- x_stk_test$year
    y_stk_test$infl_factor <- x_stk_test$infl_factor
    y_stk_test             <- preprocess_Y_data(y_stk_test)
    y_pp                   <- y_stk_test$claim_amount_inf_cap
    y_freq                 <- y_pp
    y_freq[y_freq>0]       <- 1
    
    #removing infl factor and year from x
    x_stk_test <- x_stk_test %>% select(-infl_factor)
    
    #prediction for level 1 learners, on the left-out fold
    pred_claims = predict_expected_claim_STACK(tempmodel, 
                                               x_stk_test, 
                                               leakage_blocker=TRUE,
                                               golden_feature = golden_feature,
                                               fit_lgb    = fit_lgb,
                                               fit_xgb    = fit_xgb,
                                               fit_glmnet = fit_glmnet,
                                               fit_glm    = fit_glm,
                                               fit_gam    = fit_gam,
                                               fit_cat    = fit_cat,
                                               fit_rf     = fit_rf)
    pred_claims_freq  <- pred_claims[[1]]
    pred_claims_sev   <- pred_claims[[2]]
    pred_claims_lcost <- pred_claims[[3]]
    
    
    stk_train_pred_freq[[i]]  = cbind(pred_claims_freq,  y_freq = y_freq)
    stk_train_pred_sev[[i]]   = cbind(pred_claims_sev,   y_pp = y_pp)
    stk_train_pred_lcost[[i]] = cbind(pred_claims_lcost, y_pp = y_pp)    
    
    
    model_stk_train[[i]] = tempmodel
    cat("                                                                \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$   PREDICTING :: DONE   $$$$$$$$$$$$$$$$$$$$ \n")
    cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
    cat("                                                               \n")
  } #end loop of 5 folds
  
  
  # Compiling all predictions from level 1 learners.
  
   xy_meta_train_freq  = rbind(stk_train_pred_freq[[1]], 
                               stk_train_pred_freq[[2]],
                               stk_train_pred_freq[[3]],
                               stk_train_pred_freq[[4]],
                               stk_train_pred_freq[[5]])
   
   xy_meta_train_sev   = rbind(stk_train_pred_sev[[1]], 
                               stk_train_pred_sev[[2]],
                               stk_train_pred_sev[[3]],
                               stk_train_pred_sev[[4]],
                               stk_train_pred_sev[[5]])
   
   xy_meta_train_lcost = rbind(stk_train_pred_lcost[[1]], 
                               stk_train_pred_lcost[[2]],
                               stk_train_pred_lcost[[3]],
                               stk_train_pred_lcost[[4]],
                               stk_train_pred_lcost[[5]])
   
   
  
  #xy_meta_train = rbind(stk_train_pred[[1]]) 
  
  xy_meta_train_lcost <- xy_meta_train_lcost %>% mutate(glm_fxs      = ifelse(glm_fxs > 50000, 50000, glm_fxs))
  xy_meta_train_lcost <- xy_meta_train_lcost %>% mutate(glm_lcost    = ifelse(glm_fxs > 50000, 50000, glm_lcost))
  xy_meta_train_lcost <- xy_meta_train_lcost %>% mutate(glmnet_lcost = ifelse(glm_fxs > 50000, 50000, glmnet_lcost))
 
  #xy_meta_train_lcost <- xy_meta_train_lcost %>% mutate(gam_fxs      = ifelse(gam_fxs > 50000, 50000, gam_fxs))
  #xy_meta_train_lcost <- xy_meta_train_lcost %>% mutate(gam_lcost    = ifelse(gam_fxs > 50000, 50000, gam_lcost))

    
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$   STACKING LOOP DONE   $$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")  
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$  STRUCTURE META TRAIN  $$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("                                                               \n")
  print(str(xy_meta_train_freq))
  cat("                                                               \n")
  cat("                                                               \n")
  print(str(xy_meta_train_sev))
  cat("                                                               \n")
  cat("                                                               \n")
  print(str(xy_meta_train_lcost))
  cat("                                                               \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$   SUMMARY META TRAIN   $$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n") 
  cat("                                                               \n")
  print(summary(xy_meta_train_freq))
  cat("                                                               \n")
  cat("                                                               \n")
  print(summary(xy_meta_train_sev))
  cat("                                                               \n")
  cat("                                                               \n")
  print(summary(xy_meta_train_lcost))
  cat("                                                               \n")
  
  #now training meta learner, aka level 2, aka super learner
  
  cat("                                                                \n")  
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$   META TRAINING START  $$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n") 
  cat("                                                                \n")
  print(Sys.time())
  cat("                                                                \n")
  
  if (fit_ens == TRUE){
    
     x_meta_train_lcost = xy_meta_train_lcost %>% select(-y_pp)
     y_pp = xy_meta_train_lcost %>% select(y_pp) %>% pull()
     
    
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # XGB ENSEMBLE ####
      cat("                                                \n ")
      cat("$$$ META Training XGBOOST ENSEMBLE              \n ")
      
      
    
        
      fitControl <- trainControl( method = "cv", 
                                  number = 7
                                  #allowParallel = TRUE
                                  )
      
      xgb_grid_ens <- expand.grid(
                  nrounds = seq(from = 1500, to = 3000, by = 100),
                  max_depth = c(1),
                  eta = 0.005,
                  gamma = 5.05,
                  colsample_bytree = c(0.262),
                  min_child_weight = c(6.53),
                  subsample = c(0.258))
      
      
      model_xgb_ens <- caret::train(y_pp ~ ., data=xy_meta_train_lcost,
                                  method="xgbTree", 
                                  trControl=fitControl,
                                  tuneGrid=xgb_grid_ens, 
                                  objective ='reg:tweedie', 
                                  tweedie_variance_power = 1.471429,
                                  #eval_metric = "tweedie-nloglik@1.4714", 
                                  eval_metric = "rmse",
                                  na.action = na.pass)
      
      
      cat("                                                   \n ")
      cat("$$$$ META Training XGBOOST ENSEMBLE :: Predictions \n ")
      cat("                                                   \n ")  
      
      expected_xgb_ens = predict(model_xgb_ens, 
                                 newdata = x_meta_train_lcost, 
                                 type = "raw", na.action=na.pass)
      expected_xgb_ens[expected_xgb_ens<0] <- 0

      print(summary(expected_xgb_ens)) 
      
      
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # NEURAL NET ENSEMBLE ####
      
      cat("                                                 \n ")
      cat("$$$ META Training NEURAL NETWORK ENSEMBLE        \n ")
      
      maxs <- apply(xy_meta_train_lcost, 2, max) 
      mins <- apply(xy_meta_train_lcost, 2, min)
      
      xy_scaled <- as.data.frame(scale(xy_meta_train_lcost, center = mins, scale = maxs - mins))
      
      trainX <- xy_scaled[, -ncol(xy_scaled)]
      trainY <- xy_scaled$y_pp
      
      fitControl <- trainControl(method = "cv", number = 3 ) #, allowParallel = TRUE)
      
      ann_tuneGrid <-  expand.grid(size  = c(3, 4, 5, 6),
                                   decay = c(0.00100, 0.00050, 0.00010, 0.00005) )
      
      model_ann_ens <- train(trainX, trainY,
                                  method = "nnet",
                                  trControl = fitControl,
                                  tuneGrid  = ann_tuneGrid,
                                  #trace=TRUE, 
                                  maxit=500, 
                                  linout = TRUE)
      
      
      cat("                                                          \n ")
      cat("$$$$ META Training NEURAL NETWORK ENSEMBLE :: Predictions \n ")
      cat("                                                          \n ")  
      
      expected_ann_ens_scaled <- predict(model_ann_ens, trainX)
      #expected_ann_ens = expected_ann_ens_scaled*50000
      #in case the max isn't 50,000...
      expected_ann_ens = expected_ann_ens_scaled*(max(xy_meta_train_lcost$y_pp))
      
      expected_ann_ens[expected_ann_ens<0] <- 0
      
      print(summary(expected_ann_ens))

      
      
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # RANDOM FOREST ENSEMBLE ####
      cat("                                            \n ")
      cat("$$$ META Training RANDOM FOREST ENSEMBLE         \n ")
      
      model_rf_ens <- ranger(
                              formula         = y_pp ~ ., 
                              data            = xy_meta_train_lcost, 
                              num.trees       = 110,
                              mtry            = 1,
                              min.node.size   = 4,
                              replace         = FALSE,
                              sample.fraction = 0.6,
                              splitrule       = "maxstat",
                              #verbose         = TRUE,
                              seed            = seed,
                              importance      = "impurity",
                              #respect.unordered.factors = 'order',
                              num.threads     = 8
                            )
      
      cat("                                                          \n ")
      cat("$$$$ META Training RANDOM FOREST ENSEMBLE :: Predictions  \n ")
      cat("                                                          \n ")  
      
      expected_rf_ens = predict(model_rf_ens, data=x_meta_train_lcost, type="response")$predictions
      expected_rf_ens[expected_rf_ens<0] <- 0
      
      print(summary(expected_rf_ens))
      
      
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # LightGBM ENSEMBLE ####
      cat("                                            \n ")
      cat("$$$ META Training LightGBM                  \n ")
      
      
      x_train <- xy_meta_train_lcost %>% select(-y_pp)
      y_train <- xy_meta_train_lcost %>% select(y_pp) %>% pull()
      
      x_train_sparse = Matrix(as.matrix(x_train),sparse=TRUE)
      dtrain = lgb.Dataset(data=x_train_sparse, label=y_train)
      
      cat("+++ Cross-Validation to find best number of iterations +++    \n")
      bstCV <- lgb.cv(    data = dtrain
                          , num_leaves = 4
                          , max_depth = 2
                          , min_data_in_leaf = 40
                          , min_sum_hessian_in_leaf = 80
                          , feature_fraction = 0.9
                          , bagging_fraction = 0.8
                          , bagging_freq = 3
                          , lambda_l1 = 0.2
                          , lambda_l2 = 0.1
                          , min_gain_to_split = 0.1
                          , learning_rate = 0.01
                          , num_iterations = 5000
                          , nthread = 8L
                          , eval = c("rmse")
                          , objective = "tweedie"
                          , tweedie_variance_power = 1.4719
                          , early_stopping_round = 20
                          , eval_freq = 20
                          , nfold = 5
                          , stratified = TRUE
                          , verbose = -1
                          #, verbose_eval = FALSE
      )
      # old tune... 
      cat("+++ Best number of iteration is :", bstCV$best_iter , " \n")
      
      
      model_lgb_ens <- lgb.train(
                data = dtrain
                , num_leaves = 4
                , max_depth = 2
                , min_data_in_leaf = 40
                , min_sum_hessian_in_leaf = 80
                , feature_fraction = 0.9
                , bagging_fraction = 0.8
                , bagging_freq = 3
                , lambda_l1 = 0.2
                , lambda_l2 = 0.1
                , min_gain_to_split = 0.1
                , learning_rate = 0.01 
                #, num_iterations = 550
                , num_iterations = bstCV$best_iter
                #, valids = valids
                , nthread = 8L
                , eval = "rmse"
                , objective = "tweedie"
                , tweedie_variance_power = 1.4719
                , verbose = -1
                #, verbose_eval = FALSE
      )
      
      cat("                                                          \n ")
      cat("$$$$ META Training LightGBM ENSEMBLE :: Predictions       \n ")
      cat("                                                          \n ")  
      
      expected_lgb_ens = predict(model_lgb_ens, x_train_sparse)
      expected_lgb_ens[expected_lgb_ens<0] <- 0  
      
      print(summary(expected_lgb_ens))
      
      cat("                                                          \n ")
      cat("$$$$ META Training AVERAGE ENSEMBLE :: Predictions        \n ")
      cat("                                                          \n ") 
      
      expected_avg_ens = (expected_xgb_ens + expected_ann_ens + expected_rf_ens + expected_lgb_ens) / 4
      
      print(summary(expected_avg_ens))
      
      
      
      xy_meta_train_preds <- data.frame(cbind( xgb_ens      = expected_xgb_ens,
                                               ann_ens      = expected_ann_ens,
                                               rf_ens       = expected_rf_ens,
                                               lgb_ens      = expected_lgb_ens,
                                               avg_ens      = expected_avg_ens, 
                                               y_pp         = y_pp) )
      
      
      
      
  }    
  else if(fit_ens == FALSE){
    
    cat("$$$                                                        \n")
    cat("$$$ *fit_ens* was selected as FALSE.                       \n")
    cat("$$$                                                        \n")
    cat("$$$ Fitting ensemble aborted,                              \n")
    cat("$$$ please make sure to complete the fit.                  \n")
    cat("$$$                                                        \n")  
    
    model_xgb_ens <- 123
    model_ann_ens <- 123
    model_rf_ens  <- 123
    model_lgb_ens <- 123
    
  }    
  
  cat("                                                                 \n")  
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$   META TRAINING DONE   $$$$$$$$$$$$$$$$$$$$ \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n") 
  cat("                                                                 \n")
  print(Sys.time())
  cat("                                                                 \n")
  
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #### OUTPUT ################################################################
  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  xy_meta_train = list(xy_meta_train_freq  ,
                       xy_meta_train_sev   ,
                       xy_meta_train_lcost ,
                       xy_meta_train_preds)
  
  stack_list = list(  xy_meta_train    # 
                    , model_xgb_ens   #   
                    , model_ann_ens    #
                    , model_rf_ens    # 
                    , model_lgb_ens   # 
                    )  
 
  #old   
  # stack_list = list( 
  #   model_xgb_ens    # 17
  #   , model_ann_ens    # 18
  #   , xy_meta_train )  # 19 
  
  return(stack_list)
}




#cl <- makeCluster(detectCores())
#registerDoParallel(cl)
#cl
#stopCluster(cl)
#registerDoSEQ()

#model = fit_model(Xdata, ydata)

#model

#str(model)
#summary(model$y_clean)

#model
#varImp(model)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 
#      $$$$$$$\  $$$$$$$\  $$$$$$$$\ $$$$$$$\  $$$$$$\  $$$$$$\ $$$$$$$$\ 
#      $$  __$$\ $$  __$$\ $$  _____|$$  __$$\ \_$$  _|$$  __$$\\__$$  __|
#      $$ |  $$ |$$ |  $$ |$$ |      $$ |  $$ |  $$ |  $$ /  \__|  $$ |   
#      $$$$$$$  |$$$$$$$  |$$$$$\    $$ |  $$ |  $$ |  $$ |        $$ |   
#      $$  ____/ $$  __$$< $$  __|   $$ |  $$ |  $$ |  $$ |        $$ |   
#      $$ |      $$ |  $$ |$$ |      $$ |  $$ |  $$ |  $$ |  $$\   $$ |   
#      $$ |      $$ |  $$ |$$$$$$$$\ $$$$$$$  |$$$$$$\ \$$$$$$  |  $$ |   
#      \__|      \__|  \__|\________|\_______/ \______| \______/   \__|
# 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# PREDICT ####

predict_expected_claim <- function(model, x_raw, leakage_blocker = TRUE,
                                   golden_feature = FALSE,
                                   fit_lgb    = TRUE,
                                   fit_xgb    = TRUE,
                                   fit_glmnet = TRUE,
                                   fit_glm    = TRUE,
                                   fit_gam    = FALSE,
                                   fit_cat    = TRUE,
                                   fit_rf     = TRUE,
                                   fit_ens    = TRUE){
  
  cat(" predict_expected_claim fuction \n ")
  
  model_gf <- model[[30]]
  model[[30]] <- 123
  
  # Predicting with regular models
  cat("                                                                        \n ")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ")  
  cat("||||||||                PREDICT EXPECTED CLAIM                ||||||||  \n ")
  cat("||||||||                golden feature = false                ||||||||  \n ")  
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ") 
  cat("                                                                        \n ")   
  p_reg <- predict_expected_claim_list(model, x_raw, leakage_blocker,
                                   golden_feature = FALSE,
                                   fit_lgb=fit_lgb,
                                   fit_xgb=fit_xgb,
                                   fit_glmnet=fit_glmnet,
                                   fit_glm=fit_glm,
                                   fit_gam=fit_gam,
                                   fit_cat=fit_cat,
                                   fit_rf=fit_rf,
                                   fit_ens=fit_ens)
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(model)
  gc()
  
  # Predicting with golden featured models
  
  cat("                                                                        \n ")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ")  
  cat("||||||||                PREDICT EXPECTED CLAIM                ||||||||  \n ")
  cat("||||||||                golden feature = true                 ||||||||  \n ")  
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ") 
  cat("                                                                        \n ") 
  

  p_gf  <- predict_expected_claim_list(model_gf, x_raw, leakage_blocker,
                                       golden_feature = TRUE,
                                       fit_lgb=fit_lgb,
                                       fit_xgb=fit_xgb,
                                       fit_glmnet=fit_glmnet,
                                       fit_glm=fit_glm,
                                       fit_gam=fit_gam,
                                       fit_cat=fit_cat,
                                       fit_rf=fit_rf,
                                       fit_ens=fit_ens)
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(model_gf)
  gc()
  
  golden_policies <- f_golden %>% select(id_policy) %>% distinct(id_policy)
  golden_policies$gf = 1
  
  x_raw       <- left_join(x_raw,golden_policies, by = "id_policy")
  x_raw       <- x_raw %>% mutate(gf = replace_na(gf,0))
  x_raw$p_reg <- p_reg[[1]]
  x_raw$p_gf  <- p_gf[[1]]
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(p_reg)
  rm(p_gf)
  gc()
  
  # For policies eligible for golden features, we use the gf models
  # For policies not eligible, we use the regular models
  
  x_raw <- x_raw %>% mutate(p_final = ifelse(gf == 1, p_gf, p_reg))
  
  expected_claims <- x_raw %>% select(p_final) %>% pull()
  
  return(expected_claims)
}

predict_expected_claim_list <- function(model, x_raw, leakage_blocker = TRUE,
                                        golden_feature = FALSE,
                                        fit_lgb    = TRUE,
                                        fit_xgb    = TRUE,
                                        fit_glmnet = TRUE,
                                        fit_glm    = TRUE,
                                        fit_gam    = FALSE,
                                        fit_cat    = TRUE,
                                        fit_rf     = TRUE,
                                        fit_ens    = TRUE) {

  model_xgb_freq     <- model[[1]]
  model_xgb_sev      <- model[[2]]
  model_xgb_lcost    <- model[[3]]
  model_glmnet_lcost <- model[[4]]
  model_glm_freq     <- model[[5]]
  model_glm_sev      <- model[[6]]
  model_glm_lcost    <- model[[7]]
  
  #model_gam_freq     <- model[[8]]
  #model_gam_sev      <- model[[9]]
  #model_gam_lcost    <- model[[10]]
  
  model_cat_freq     <- model[[8]]
  model_cat_sev      <- model[[9]]
  model_cat_lcost    <- model[[10]]
  model_rf_freq      <- model[[11]] 
  model_rf_sev       <- model[[12]] 
  model_rf_lcost     <- model[[13]]
  model_lgb_freq     <- model[[14]] 
  model_lgb_sev      <- model[[15]] 
  model_lgb_lcost    <- model[[16]] 
  cs_model           <- model[[17]]
  pca_model          <- model[[18]]
  #ens_models_freq    <- model[[19]] #not used for predicts
  #ens_models_sev     <- model[[20]] #not used for predicts
  #ens_models_lcost   <- model[[21]] #not used for predicts
  #inTrain            <- model[[22]] #not used for real predicts
  #xy_meta_train      <- model[[23]]
  #xy_meta_test      <- model[[24]
  model_xgb_ens      <- model[[25]] #for stacking at the end
  model_ann_ens      <- model[[26]] #for stacking at the end, if I end up using an ANN
  model_rf_ens       <- model[[27]]
  model_lgb_ens      <- model[[28]]
  #obf_train          <- model[[29]]
  
  xy_meta_train_lcost      <- model[[23]][[3]]
  
  maxs <- apply(xy_meta_train_lcost, 2, max) 
  mins <- apply(xy_meta_train_lcost, 2, min)
  
  maxs <- head(maxs,11)
  mins <- head(mins,11)
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # LGB

    cat("                                                              \n")
    cat("**** Predicting LGB **** \n")
    x_clean = preprocess_X_data(x_raw,
                                one_hot = TRUE,
                                target_encoding = TRUE,
                                leave_one_out = FALSE,
                                leakage_blocker = leakage_blocker,
                                golden_feature = golden_feature,
                                keep_vh_make_model = FALSE)   #for lgb & xgb
    
    #saving for future use
    x_clean_saved_for_rf <- x_clean
    infl <- x_clean$infl_factor
    ###
    
  if(fit_lgb == TRUE){    
    x_clean <- x_clean %>% select(-infl_factor, -year)
    #print(data.frame(colnames(x_clean)))
    print(nrow(data.frame(colnames(x_clean))))
    
    x_clean_sparse = Matrix(as.matrix(x_clean),sparse=TRUE)
    
    expected_lgb_freq  = predict(model_lgb_freq, x_clean_sparse,predict_disable_shape_check=FALSE)
    expected_lgb_sev   = predict(model_lgb_sev, x_clean_sparse,predict_disable_shape_check=FALSE)
    expected_lgb_lcost = predict(model_lgb_lcost, x_clean_sparse,predict_disable_shape_check=FALSE)
    
    expected_lgb_sev[expected_lgb_sev>50000] <- 50000
    
    expected_lgb_fxs = expected_lgb_freq * expected_lgb_sev / infl
    expected_lgb_lcost = expected_lgb_lcost / infl
    
    expected_lgb_sev[expected_lgb_sev<0]     <- 0
    expected_lgb_fxs[expected_lgb_fxs<0]     <- 0
    expected_lgb_lcost[expected_lgb_lcost<0] <- 0
  }
  else if(fit_lgb == FALSE){
    expected_lgb_freq  = rep(0.10, length(infl))
    expected_lgb_sev   = rep(1000, length(infl)) 
    expected_lgb_fxs   = rep(101,  length(infl))
    expected_lgb_lcost = rep(100,  length(infl)) 
  }
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # XGB
  cat("                                                              \n")
  cat("**** Predicting XGB **** \n")
  #x_clean = preprocess_X_data(x_raw,
  #                            one_hot = TRUE,
  #                            target_encoding = TRUE,
  #                            leave_one_out = FALSE,
  #                            leakage_blocker = leakage_blocker,
  #                            keep_vh_make_model = FALSE)   #for xgb
  #
  #x_clean_saved_for_rf <- x_clean
  if(fit_xgb == TRUE){     
    expected_xgb_freq  = predict(model_xgb_freq,  newdata = x_clean, type = "prob", na.action=na.pass)[,"Y"]
    expected_xgb_sev   = predict(model_xgb_sev,   newdata = x_clean, type = "raw", na.action=na.pass)
    expected_xgb_lcost = predict(model_xgb_lcost, newdata = x_clean, type = "raw", na.action=na.pass)
  
    expected_xgb_sev[expected_xgb_sev>50000] <- 50000
    
    expected_xgb_fxs = expected_xgb_freq * expected_xgb_sev / infl
    expected_xgb_lcost = expected_xgb_lcost / infl
  
    expected_xgb_sev[expected_xgb_sev<0]     <- 0
    expected_xgb_fxs[expected_xgb_fxs<0]     <- 0
    expected_xgb_lcost[expected_xgb_lcost<0] <- 0
  }
  else if (fit_xgb == FALSE){

    expected_xgb_freq  = rep(0.11, length(infl))
    expected_xgb_sev   = rep(1100, length(infl)) 
    expected_xgb_fxs   = rep(111,  length(infl))
    expected_xgb_lcost = rep(110,  length(infl))
    
  }

  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GLMNET
  cat("                                                              \n")
  cat("**** Predicting GLMNET **** \n")

  x_clean = preprocess_X_data(x_raw,
                              one_hot = TRUE,
                              target_encoding = TRUE,
                              leave_one_out = TRUE,
                              leakage_blocker = leakage_blocker,
                              golden_feature = golden_feature,
                              keep_vh_make_model = FALSE)   #for glmnet

  if(fit_glmnet == TRUE){ 
    
    expected_glmnet_lcost = predict(model_glmnet_lcost, newdata = x_clean, type = "raw")
  
    infl <- x_clean$infl_factor
    expected_glmnet_lcost = expected_glmnet_lcost / infl
  
    expected_glmnet_lcost[expected_glmnet_lcost<0] <- 0
    expected_glmnet_lcost[expected_glmnet_lcost>50000] <- 50000
 
  }
  else if (fit_glmnet == FALSE){

    expected_glmnet_lcost = rep(120,  length(infl))
    
  }  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GLM
  cat("                                                              \n")
  cat("**** Predicting GLM **** \n")
  
  #x_clean = preprocess_X_data(x_raw,
  #                            one_hot = TRUE,
  #                            target_encoding = TRUE,
  #                            leave_one_out = TRUE)   #for glm
  
  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  x_clean <- apply_cs_pca(x_clean, cs_model, pca_model)
  x_clean <- trim_variables(x_clean)
  
  if (fit_glm == TRUE) {    
  
    expected_glm_freq  = predict(model_glm_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_glm_sev   = predict(model_glm_sev,   newdata = x_clean, type = "raw" )
    expected_glm_lcost = predict(model_glm_lcost, newdata = x_clean, type = "raw")
    
    expected_glm_sev[expected_glm_sev>50000] <- 50000
    
    expected_glm_fxs   = expected_glm_freq * expected_glm_sev / infl
    expected_glm_lcost = expected_glm_lcost / infl
    
    expected_glm_sev[expected_glm_sev<0]     <- 0  
    expected_glm_fxs[expected_glm_fxs<0]     <- 0
    expected_glm_lcost[expected_glm_lcost<0] <- 0
  } 
  else if (fit_glm == FALSE){

      expected_glm_freq  = rep(0.13, length(infl))
      expected_glm_sev   = rep(1300, length(infl)) 
      expected_glm_fxs   = rep(131,  length(infl))
      expected_glm_lcost = rep(130,  length(infl))
      
  }   
    
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GAM
  cat("                                                              \n")
  cat("**** Predicting GAM **** \n")
  
  if (fit_gam == TRUE) {    
    
    expected_gam_freq  = predict(model_gam_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_gam_sev   = predict(model_gam_sev,   newdata = x_clean, type = "raw" )
    expected_gam_lcost = predict(model_gam_lcost, newdata = x_clean, type = "raw")
    
    expected_gam_sev[expected_gam_sev>50000] <- 50000
    
    expected_gam_fxs   = expected_gam_freq * expected_gam_sev / infl
    expected_gam_lcost = expected_gam_lcost / infl
    
    expected_gam_sev[expected_gam_sev<0]     <- 0  
    expected_gam_fxs[expected_gam_fxs<0]     <- 0
    expected_gam_lcost[expected_gam_lcost<0] <- 0
  } 
  else if (fit_gam == FALSE){
    
    expected_gam_freq  = rep(0.133, length(infl))
    expected_gam_sev   = rep(1330, length(infl)) 
    expected_gam_fxs   = rep(133,  length(infl))
    expected_gam_lcost = rep(132,  length(infl))
    
  }   

  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # CATBOOST
  cat("                                                              \n")
  cat("**** Predicting CATBOOST **** \n")


  x_clean = preprocess_X_data(x_raw,
                              one_hot = FALSE,
                              target_encoding = FALSE,
                              leakage_blocker = leakage_blocker,
                              golden_feature = golden_feature,
                              keep_vh_make_model = TRUE)

  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]

  x_clean_pool <- catboost.load_pool(data = x_clean)

  if (fit_cat == TRUE){ 
    
    expected_cat_freq  = catboost.predict(model_cat_freq, x_clean_pool,  prediction_type='Probability')
    expected_cat_sev   = catboost.predict(model_cat_sev, x_clean_pool)
    expected_cat_lcost = catboost.predict(model_cat_lcost, x_clean_pool,  prediction_type='Exponent')
  
    expected_cat_sev[expected_cat_sev>50000] <- 50000
  
    expected_cat_fxs   = expected_cat_freq * expected_cat_sev / infl
    expected_cat_lcost = expected_cat_lcost / infl
  
    expected_cat_sev[expected_cat_sev<0]     <- 0
    expected_cat_fxs[expected_cat_fxs<0]     <- 0
    expected_cat_lcost[expected_cat_lcost<0] <- 0

  }
  else if (fit_cat == FALSE){

    expected_cat_freq  = rep(0.14, length(infl))
    expected_cat_sev   = rep(1400, length(infl)) 
    expected_cat_fxs   = rep(141,  length(infl))
    expected_cat_lcost = rep(140,  length(infl))
      
  }  

  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # RANDOM FOREST 
  cat("                                                              \n")
  cat("**** Predicting RANDOM FOREST **** \n")   
  
  
  #x_clean = preprocess_X_data(x_raw, 
  #                            one_hot = TRUE, 
  #                            target_encoding = TRUE)   
  
  x_clean <- x_clean_saved_for_rf
  
  if (golden_feature == TRUE) {
    x_gf <- x_clean %>% select(F_claim_cnt, 
                               F_claim_cnt_c, 
                               F_claim_amt_c,
                               F_claim_amt_avg,
                               F_cov)
  }
  
  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  #x_trim = trim_variables_broad_leak(x_clean)
  x_trim = trim_variables_broad(x_clean)
  
  
  if (golden_feature == TRUE) {
    x_trim = cbind(x_trim,x_gf)
  }
  
  if (fit_rf == TRUE) {  
    
    ef  = predict(model_rf_freq,  data=x_trim, type="response")
    es  = predict(model_rf_sev,   data=x_trim, type="response")  
    elc = predict(model_rf_lcost, data=x_trim, type="response")
    
    expected_rf_freq  = ef$predictions[,"Y"]
    expected_rf_sev   = es$predictions
    
    expected_rf_sev[expected_rf_sev>50000] <- 50000
    
    
    expected_rf_fxs   = expected_rf_freq * expected_rf_sev / infl
    expected_rf_lcost = elc$predictions / infl
  
    expected_rf_sev[expected_rf_sev<0]     <- 0   
    expected_rf_fxs[expected_rf_fxs<0]     <- 0  
    expected_rf_lcost[expected_rf_lcost<0] <- 0
  }
  
  else if (fit_rf == FALSE){
    
    expected_rf_freq  = rep(0.15, length(infl))
    expected_rf_sev   = rep(1500, length(infl)) 
    expected_rf_fxs   = rep(151,  length(infl))
    expected_rf_lcost = rep(150,  length(infl))
    
  } 
  

  #patch because ranger returned NaN once...
  expected_rf_lcost[is.nan(expected_rf_lcost)] <- expected_rf_fxs
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Weights 
  cat("                                                   \n")
  cat("************************************************** \n")
  cat("**** Predicting Ensemble & Stacking           **** \n")
  
  # weights determined by 2M random sampling on 4 folds of holdout data.
  w_xgb_fxs      = 30/100
  w_xgb_lcost    = 0/100
  w_glmnet_lcost = 3/100
  w_glm_fxs      = 0/100
  w_glm_lcost    = 20/100
  w_gam_fxs      = 0/100
  w_gam_lcost    = 0/100
  w_cat_fxs      = 8/100
  w_cat_lcost    = 2/100
  w_rf_fxs       = 0/100
  w_rf_lcost     = 17/100
  w_lgb_fxs      = 10/100
  w_lgb_lcost    = 10/100
  
  # legacy when I used a weighted avg...
  expected_claims_wgt =  (w_xgb_fxs      * expected_xgb_fxs      +
                          w_xgb_lcost    * expected_xgb_lcost    +
                          w_glmnet_lcost * expected_glmnet_lcost +
                          w_glm_fxs      * expected_glm_fxs      +
                          w_glm_lcost    * expected_glm_lcost    +
                          #w_gam_fxs      * expected_gam_fxs      +
                          #w_gam_lcost    * expected_gam_lcost    +                            
                          w_cat_fxs      * expected_cat_fxs      +
                          w_cat_lcost    * expected_cat_lcost    +
                          w_rf_fxs       * expected_rf_fxs       +
                          w_rf_lcost     * expected_rf_lcost     +
                          w_lgb_fxs      * expected_lgb_fxs      +
                          w_lgb_lcost    * expected_lgb_lcost       )

  
  ens_models_lcost <- data.frame(cbind(xgb_fxs      = expected_xgb_fxs,
                                       xgb_lcost    = expected_xgb_lcost,
                                       glmnet_lcost = expected_glmnet_lcost,
                                       glm_fxs      = expected_glm_fxs,
                                       glm_lcost    = expected_glm_lcost,
                                       #gam_fxs      = expected_gam_fxs,
                                       #gam_lcost    = expected_gam_lcost,
                                       cat_fxs      = expected_cat_fxs,
                                       cat_lcost    = expected_cat_lcost,
                                       rf_fxs       = expected_rf_fxs,
                                       rf_lcost     = expected_rf_lcost,
                                       lgb_fxs      = expected_lgb_fxs, 
                                       lgb_lcost    = expected_lgb_lcost ))
  
  cat("************************************************** \n")
  cat("***    all level 1 models predictions          *** \n")
  cat("                                             \n")
  print(summary(ens_models_lcost))
   
  
  if (fit_ens == TRUE){
      cat("                               \n")
      cat("************************************************** \n")
      cat("****    Predicting level 2                    **** \n")
      cat("                               \n")      
      # xgb stacker
      cat("                        \n")  
      cat("****** XGB stacker      \n")
      cat("                        \n")
      expected_xgb_ens = predict(model_xgb_ens, newdata = ens_models_lcost, type = "raw", na.action=na.pass)
      expected_xgb_ens[expected_xgb_ens<0] <- 0
      
      
      print(summary(expected_xgb_ens))
      
      # ann stacker
    
      cat("                        \n")  
      cat("****** ANN stacker      \n")
      cat("                        \n")
      #maxs = c(669.18542070778, 754.614823069149, 893.444636204407, 3831.04999756495, 2925.32159376627, 745.207472219029, 432.818818231591, 930.287962442134, 573.590011798459, 590.943832014846, 643.592749956324)
      #mins = c(3.92923777919976, 0.059137936326479, 0, 1.49429953760516e-09, 1.06825264439808e-09, 4.52160803423322, 5.1441626967171, 0.108924257037318, 4.44188543314681, 3.52420295633553, 0.690862778619006)
      ens_models_lcost_scaled <- as.data.frame(scale(ens_models_lcost, center = mins, scale = maxs - mins))
      expected_ann_ens_scaled <- predict(model_ann_ens, ens_models_lcost_scaled)  
      #expected_ann_ens = expected_ann_ens_scaled*50000
      expected_ann_ens = expected_ann_ens_scaled*(max(xy_meta_train_lcost$y_pp))
      expected_ann_ens[expected_ann_ens<0] <- 0
      
      print(summary(expected_ann_ens))
      
      # rf stacker
      cat("                        \n")  
      cat("****** RF  stacker      \n")
      cat("                        \n")  
      expected_rf_ens = predict(model_rf_ens, data=ens_models_lcost, type="response")$predictions
      expected_rf_ens[expected_rf_ens<0] <- 0
      
      print(summary(expected_rf_ens))
      
      #lgb stacker
      cat("                        \n")  
      cat("****** LGB stacker      \n")
      cat("                        \n")   
      ens_models_lcost_sparse = Matrix(as.matrix(ens_models_lcost),sparse=TRUE)
      expected_lgb_ens = predict(model_lgb_ens, ens_models_lcost_sparse)
      expected_lgb_ens[expected_lgb_ens<0] <- 0  
      
      print(summary(expected_lgb_ens))
      
      #averaging the stackers
      cat("                        \n")  
      cat("****** Average of stackers      \n")
      cat("                        \n")      
      expected_avg_ens = (expected_xgb_ens + expected_ann_ens + expected_rf_ens + expected_lgb_ens) / 4
  
      print(summary(expected_avg_ens))
      
  }
  else if(fit_ens == FALSE){
      cat("***** fit_ens set to FALSE...  \n ")
      cat("***** using weighted prediction instead of ensemble...  \n ")
      expected_avg_ens = expected_claims_wgt
  }
  
  
  expected_avg_freq = (expected_xgb_freq + 
                         expected_glm_freq + 
                         expected_cat_freq + 
                         expected_rf_freq +
                         expected_lgb_freq) / 5
 
  expected_avg_sev = (expected_xgb_sev + 
                         expected_glm_sev + 
                         expected_cat_sev + 
                         expected_rf_sev +
                         expected_lgb_sev) / 5
  
  expected_avg_fxs = (expected_xgb_fxs + 
                        expected_glm_fxs + 
                        expected_cat_fxs + 
                        expected_rf_fxs +
                        expected_lgb_fxs) / 5
  
  expected_avg_lcost = (expected_xgb_lcost + 
                          expected_glmnet_lcost +
                          expected_glm_lcost + 
                          expected_cat_lcost + 
                          expected_rf_lcost +
                          expected_lgb_lcost) / 6
  
  
  expected_freq_list <- list(expected_xgb_freq,
                             expected_glm_freq,
                             expected_cat_freq,
                             expected_rf_freq,
                             expected_lgb_freq,
                             expected_avg_freq)
  
  expected_sev_list <- list(expected_xgb_sev,
                             expected_glm_sev,
                             expected_cat_sev,
                             expected_rf_sev,
                             expected_lgb_sev,
                             expected_avg_sev)
  
  expected_lcost_list <- list(expected_xgb_fxs,
                              expected_xgb_lcost,
                              expected_glmnet_lcost,
                              expected_glm_fxs,
                              expected_glm_lcost,
                              expected_cat_fxs,
                              expected_cat_lcost,
                              expected_rf_fxs,
                              expected_rf_lcost,
                              expected_lgb_fxs,
                              expected_lgb_lcost,
                              expected_avg_fxs,
                              expected_avg_lcost) 
  
  
  expected_claims_list <- list(expected_avg_ens, 
                               expected_xgb_ens, 
                               expected_ann_ens,
                               expected_rf_ens,
                               expected_lgb_ens,
                               expected_freq_list,
                               expected_sev_list,
                               expected_lcost_list
                               )
  
  
  # ens_pool <- catboost.load_pool(data = ens_models)
  # 
  # cat_ens_tw = catboost.predict(model_cat_ens_tw, 
  #                               ens_pool, 
  #                               prediction_type='Exponent')
  # 
  # summary(cat_ens_tw)
  # cat_ens_tw[cat_ens_tw<0] = 0
  # expected_claims = cat_ens_tw
  

  
  return(expected_claims_list)  
}






predict_expected_claim_STACK <- function(model, x_raw, leakage_blocker = TRUE,
                                         golden_feature = FALSE,
                                         fit_lgb    = TRUE,
                                         fit_xgb    = TRUE,
                                         fit_glmnet = TRUE,
                                         fit_glm    = TRUE,
                                         fit_gam    = FALSE,
                                         fit_cat    = TRUE,
                                         fit_rf     = TRUE ) {
  
  model_xgb_freq     <- model[[1]]
  model_xgb_sev      <- model[[2]]
  model_xgb_lcost    <- model[[3]]
  model_glmnet_lcost <- model[[4]]
  model_glm_freq     <- model[[5]]
  model_glm_sev      <- model[[6]]
  model_glm_lcost    <- model[[7]]
  #model_gam_freq     <- model[[8]]
  #model_gam_sev      <- model[[9]]
  #model_gam_lcost    <- model[[10]]  
  model_cat_freq     <- model[[8]]
  model_cat_sev      <- model[[9]]
  model_cat_lcost    <- model[[10]]
  model_rf_freq      <- model[[11]] 
  model_rf_sev       <- model[[12]] 
  model_rf_lcost     <- model[[13]] 
  model_lgb_freq     <- model[[14]] 
  model_lgb_sev      <- model[[15]] 
  model_lgb_lcost    <- model[[16]]   
  cs_model           <- model[[17]]
  pca_model          <- model[[18]]
#  ens_models_freq    <- model[[19]] #not used for predicts
#  ens_models_sev     <- model[[20]] 
#  ens_models_lcost   <- model[[21]] 
#  inTrain            <- model[[22]] #not used for real predicts
  
  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # LGB
  
  cat("                                                              \n")
  cat("**** Predicting LGB **** \n")
  x_clean = preprocess_X_data(x_raw,
                              one_hot = TRUE,
                              target_encoding = TRUE,
                              leave_one_out = FALSE,
                              leakage_blocker = leakage_blocker,
                              golden_feature = golden_feature,
                              keep_vh_make_model = FALSE)   #for lgb & xgb
  
  #saving for future use
  x_clean_saved_for_rf <- x_clean
  infl <- x_clean$infl_factor
  ###
  
  if(fit_lgb == TRUE){    
    x_clean <- x_clean %>% select(-infl_factor, -year)
    
    x_clean_sparse = Matrix(as.matrix(x_clean),sparse=TRUE)
    
    expected_lgb_freq  = predict(model_lgb_freq, x_clean_sparse)
    expected_lgb_sev   = predict(model_lgb_sev, x_clean_sparse)
    expected_lgb_lcost = predict(model_lgb_lcost, x_clean_sparse)
    
    expected_lgb_sev[expected_lgb_sev>50000] <- 50000
    
    expected_lgb_fxs = expected_lgb_freq * expected_lgb_sev / infl
    expected_lgb_lcost = expected_lgb_lcost / infl
    
    expected_lgb_sev[expected_lgb_sev<0]     <- 0
    expected_lgb_fxs[expected_lgb_fxs<0]     <- 0
    expected_lgb_lcost[expected_lgb_lcost<0] <- 0
  }
  else if(fit_lgb == FALSE){
    expected_lgb_freq  = rep(0.10, length(infl))
    expected_lgb_sev   = rep(1000, length(infl)) 
    expected_lgb_fxs   = rep(101,  length(infl))
    expected_lgb_lcost = rep(100,  length(infl)) 
  }
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # XGB
  cat("                                                              \n")
  cat("**** Predicting XGB **** \n")
  #x_clean = preprocess_X_data(x_raw,
  #                            one_hot = TRUE,
  #                            target_encoding = TRUE,
  #                            leave_one_out = FALSE,
  #                            leakage_blocker = leakage_blocker,
  #                            keep_vh_make_model = FALSE)   #for xgb
  #
  #x_clean_saved_for_rf <- x_clean
  if(fit_xgb == TRUE){     
    expected_xgb_freq  = predict(model_xgb_freq,  newdata = x_clean, type = "prob", na.action=na.pass)[,"Y"]
    expected_xgb_sev   = predict(model_xgb_sev,   newdata = x_clean, type = "raw", na.action=na.pass)
    expected_xgb_lcost = predict(model_xgb_lcost, newdata = x_clean, type = "raw", na.action=na.pass)
    
    expected_xgb_sev[expected_xgb_sev>50000] <- 50000
    
    expected_xgb_fxs = expected_xgb_freq * expected_xgb_sev / infl
    expected_xgb_lcost = expected_xgb_lcost / infl
    
    expected_xgb_sev[expected_xgb_sev<0]     <- 0
    expected_xgb_fxs[expected_xgb_fxs<0]     <- 0
    expected_xgb_lcost[expected_xgb_lcost<0] <- 0
  }
  else if (fit_xgb == FALSE){
    
    expected_xgb_freq  = rep(0.11, length(infl))
    expected_xgb_sev   = rep(1100, length(infl)) 
    expected_xgb_fxs   = rep(111,  length(infl))
    expected_xgb_lcost = rep(110,  length(infl))
    
  }
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GLMNET
  cat("                                                              \n")
  cat("**** Predicting GLMNET **** \n")
  
  x_clean = preprocess_X_data(x_raw,
                              one_hot = TRUE,
                              target_encoding = TRUE,
                              leave_one_out = TRUE,
                              leakage_blocker = leakage_blocker,
                              golden_feature = golden_feature,
                              keep_vh_make_model = FALSE)   #for glmnet
  
  if(fit_glmnet == TRUE){ 
    
    expected_glmnet_lcost = predict(model_glmnet_lcost, newdata = x_clean, type = "raw")
    
    infl <- x_clean$infl_factor
    expected_glmnet_lcost = expected_glmnet_lcost / infl
    
    expected_glmnet_lcost[expected_glmnet_lcost<0] <- 0
    expected_glmnet_lcost[expected_glmnet_lcost>50000] <- 50000
    
  }
  else if (fit_glmnet == FALSE){
    
    expected_glmnet_lcost = rep(120,  length(infl))
    
  }  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GLM
  cat("                                                              \n")
  cat("**** Predicting GLM **** \n")
  
  #x_clean = preprocess_X_data(x_raw,
  #                            one_hot = TRUE,
  #                            target_encoding = TRUE,
  #                            leave_one_out = TRUE)   #for glm
  
  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  x_clean <- apply_cs_pca(x_clean, cs_model, pca_model)
  x_clean <- trim_variables(x_clean)
  
  if (fit_glm == TRUE) {    
    
    expected_glm_freq  = predict(model_glm_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_glm_sev   = predict(model_glm_sev,   newdata = x_clean, type = "raw" )
    expected_glm_lcost = predict(model_glm_lcost, newdata = x_clean, type = "raw")
    
    expected_glm_sev[expected_glm_sev>50000] <- 50000
    
    expected_glm_fxs   = expected_glm_freq * expected_glm_sev / infl
    expected_glm_lcost = expected_glm_lcost / infl
    
    expected_glm_sev[expected_glm_sev<0]     <- 0  
    expected_glm_fxs[expected_glm_fxs<0]     <- 0
    expected_glm_lcost[expected_glm_lcost<0] <- 0
  } 
  else if (fit_glm == FALSE){
    
    expected_glm_freq  = rep(0.13, length(infl))
    expected_glm_sev   = rep(1300, length(infl)) 
    expected_glm_fxs   = rep(131,  length(infl))
    expected_glm_lcost = rep(130,  length(infl))
    
  }   
  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # GAM
  cat("                                                              \n")
  cat("**** Predicting GAM **** \n")
  
  
  
  if (fit_gam == TRUE) {    
    
    expected_gam_freq  = predict(model_gam_freq,  newdata = x_clean, type = "prob")[,"Y"]
    expected_gam_sev   = predict(model_gam_sev,   newdata = x_clean, type = "raw" )
    expected_gam_lcost = predict(model_gam_lcost, newdata = x_clean, type = "raw")
    
    expected_gam_sev[expected_gam_sev>50000] <- 50000
    
    expected_gam_fxs   = expected_gam_freq * expected_gam_sev / infl
    expected_gam_lcost = expected_gam_lcost / infl
    
    expected_gam_sev[expected_gam_sev<0]     <- 0  
    expected_gam_fxs[expected_gam_fxs<0]     <- 0
    expected_gam_lcost[expected_gam_lcost<0] <- 0
  } 
  else if (fit_gam == FALSE){
    
    expected_gam_freq  = rep(0.13, length(infl))
    expected_gam_sev   = rep(1300, length(infl)) 
    expected_gam_fxs   = rep(131,  length(infl))
    expected_gam_lcost = rep(130,  length(infl))
    
  }   
  
  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # CATBOOST
  cat("                                                              \n")
  cat("**** Predicting CATBOOST **** \n")
  
  
  x_clean = preprocess_X_data(x_raw,
                              one_hot = FALSE,
                              target_encoding = FALSE,
                              leakage_blocker = leakage_blocker,
                              golden_feature = golden_feature,
                              keep_vh_make_model = TRUE)
  
  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  x_clean_pool <- catboost.load_pool(data = x_clean)
  
  if (fit_cat == TRUE){ 
    
    expected_cat_freq  = catboost.predict(model_cat_freq, x_clean_pool,  prediction_type='Probability')
    expected_cat_sev   = catboost.predict(model_cat_sev, x_clean_pool)
    expected_cat_lcost = catboost.predict(model_cat_lcost, x_clean_pool,  prediction_type='Exponent')
    
    expected_cat_sev[expected_cat_sev>50000] <- 50000
    
    expected_cat_fxs   = expected_cat_freq * expected_cat_sev / infl
    expected_cat_lcost = expected_cat_lcost / infl
    
    expected_cat_sev[expected_cat_sev<0]     <- 0
    expected_cat_fxs[expected_cat_fxs<0]     <- 0
    expected_cat_lcost[expected_cat_lcost<0] <- 0
    
  }
  else if (fit_cat == FALSE){
    
    expected_cat_freq  = rep(0.14, length(infl))
    expected_cat_sev   = rep(1400, length(infl)) 
    expected_cat_fxs   = rep(141,  length(infl))
    expected_cat_lcost = rep(140,  length(infl))
    
  }  
  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # RANDOM FOREST 
  cat("                                                              \n")
  cat("**** Predicting RANDOM FOREST **** \n")   
  
  
  #x_clean = preprocess_X_data(x_raw, 
  #                            one_hot = TRUE, 
  #                            target_encoding = TRUE)   
  
  x_clean <- x_clean_saved_for_rf
  
  if (golden_feature == TRUE) {
    x_gf <- x_clean %>% select(F_claim_cnt, 
                               F_claim_cnt_c, 
                               F_claim_amt_c,
                               F_claim_amt_avg,
                               F_cov)
  }
  
  infl <- x_clean$infl_factor
  x_clean <- x_clean[ , !(names(x_clean)=="year")]
  x_clean <- x_clean[ , !(names(x_clean)=="infl_factor")]
  
  #x_trim = trim_variables_broad_leak(x_clean)
  x_trim = trim_variables_broad(x_clean)
  
  if (golden_feature == TRUE) {
    x_trim = cbind(x_trim,x_gf)
  }
  
  if (fit_rf == TRUE) {  
    
    ef  = predict(model_rf_freq,  data=x_trim, type="response")
    es  = predict(model_rf_sev,   data=x_trim, type="response")  
    elc = predict(model_rf_lcost, data=x_trim, type="response")
    
    expected_rf_freq  = ef$predictions[,"Y"]
    expected_rf_sev   = es$predictions
    
    expected_rf_sev[expected_rf_sev>50000] <- 50000
    
    
    expected_rf_fxs   = expected_rf_freq * expected_rf_sev / infl
    expected_rf_lcost = elc$predictions / infl
    
    expected_rf_sev[expected_rf_sev<0]     <- 0   
    expected_rf_fxs[expected_rf_fxs<0]     <- 0  
    expected_rf_lcost[expected_rf_lcost<0] <- 0
  }
  
  else if (fit_rf == FALSE){
    
    expected_rf_freq  = rep(0.15, length(infl))
    expected_rf_sev   = rep(1500, length(infl)) 
    expected_rf_fxs   = rep(151,  length(infl))
    expected_rf_lcost = rep(150,  length(infl))
    
  } 
  

  
  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # All predictions 
  cat("                                                              \n")
  cat("**** Compiling All Predictions **** \n")

  expected_claims_freq = as.data.frame(cbind(xgb_freq   = expected_xgb_freq,
                                            glm_freq    = expected_glm_freq,
                                            #gam_freq    = expected_gam_freq,
                                            cat_freq    = expected_cat_freq,
                                            rf_freq     = expected_rf_freq,
                                            lgb_freq    = expected_lgb_freq ))

  expected_claims_sev = as.data.frame(cbind(xgb_sev   = expected_xgb_sev,
                                             glm_sev    = expected_glm_sev,
                                             #gam_sev    = expected_gam_sev,
                                             cat_sev    = expected_cat_sev,
                                             rf_sev     = expected_rf_sev,
                                             lgb_sev    = expected_lgb_sev ))
    
  expected_claims_lcost = as.data.frame(cbind(xgb_fxs      = expected_xgb_fxs,
                                            xgb_lcost    = expected_xgb_lcost,
                                            glmnet_lcost = expected_glmnet_lcost,
                                            glm_fxs      = expected_glm_fxs,
                                            glm_lcost    = expected_glm_lcost,
                                            #gam_fxs      = expected_gam_fxs,
                                            #gam_lcost    = expected_gam_lcost,                                           
                                            cat_fxs      = expected_cat_fxs,
                                            cat_lcost    = expected_cat_lcost,
                                            rf_fxs       = expected_rf_fxs,
                                            rf_lcost     = expected_rf_lcost,
                                            lgb_fxs      = expected_lgb_fxs,
                                            lgb_lcost    = expected_lgb_lcost ))
  

  expected_claims_all <- list(expected_claims_freq ,
                              expected_claims_sev  ,
                              expected_claims_lcost)
  
  
  
  return(expected_claims_all)  
}









predict_premium <- function(model, x_raw, leakage_blocker = TRUE,
                            golden_feature = FALSE,
                            fit_lgb    = TRUE,
                            fit_xgb    = TRUE,
                            fit_glmnet = TRUE,
                            fit_glm    = TRUE,
                            fit_gam    = FALSE,
                            fit_cat    = TRUE,
                            fit_rf     = TRUE,
                            fit_ens    = TRUE){
  
  cat(" predict_premium fuction \n ")
  
  model_gf <- model[[30]]
  model[[30]] <- 123
  
  
  cat("                                                                        \n ")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ")  
  cat("||||||||                PREDICT EXPECTED CLAIM                ||||||||  \n ")
  cat("||||||||                golden feature = false                ||||||||  \n ")  
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ") 
  cat("                                                                        \n ")
  
  # Predicting with regular models
  p_reg <- predict_expected_claim_list(model, x_raw, leakage_blocker,
                                       golden_feature = FALSE,
                                       fit_lgb=fit_lgb,
                                       fit_xgb=fit_xgb,
                                       fit_glmnet=fit_glmnet,
                                       fit_glm=fit_glm,
                                       fit_gam=fit_gam,
                                       fit_cat=fit_cat,
                                       fit_rf=fit_rf,
                                       fit_ens=fit_ens)
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(model)
  gc()
  
  
  # Predicting with golden featured models
  cat("                                                                        \n ")
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ")  
  cat("||||||||                PREDICT EXPECTED CLAIM                ||||||||  \n ")
  cat("||||||||                golden feature = true                 ||||||||  \n ")  
  cat("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  \n ") 
  cat("                                                                        \n ")

  p_gf  <- predict_expected_claim_list(model_gf, x_raw, leakage_blocker,
                                       golden_feature = TRUE,
                                       fit_lgb=fit_lgb,
                                       fit_xgb=fit_xgb,
                                       fit_glmnet=fit_glmnet,
                                       fit_glm=fit_glm,
                                       fit_gam=fit_gam,
                                       fit_cat=fit_cat,
                                       fit_rf=fit_rf,
                                       fit_ens=fit_ens)
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(model_gf)
  gc()
  
  
  golden_policies <- f_golden %>% select(id_policy) %>% distinct(id_policy)
  golden_policies$gf = 1
  
  x_raw       <- left_join(x_raw,golden_policies, by = "id_policy")
  x_raw       <- x_raw %>% mutate(gf = replace_na(gf,0))
  x_raw$p_reg <- p_reg[[1]]
  x_raw$p_gf  <- p_gf[[1]]
  
  
  # For policies eligible for golden features, we use the gf models
  # For policies not eligible, we use the regular models
  
  x_raw <- x_raw %>% mutate(p_final = ifelse(gf == 1, p_gf, p_reg))
  
  expected_claims <- x_raw %>% select(p_final) %>% pull()
  
  
  expected_freq_reg <- p_reg[[6]][[6]] #avg_freq
  expected_freq_gf  <- p_gf[[6]][[6]]  #avg_freq
  
  # REMOVING ITEMS TO TRY TO FREE UP MEMORY
  rm(p_reg)
  rm(p_gf)
  gc()
  
  freq_split = data.frame(cbind(gf    = x_raw$gf, 
                                 f_reg = expected_freq_reg ,
                                 f_gf  = expected_freq_gf   ))
  
  freq_split <- freq_split %>% mutate(f_final = ifelse(gf==1, f_gf, f_reg))
  
  gf_proportion = sum(freq_split$gf) / nrow(freq_split)
  
  e_freq = exp(freq_split$f_final)
  
  cat("                                                       \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  \n")
  cat("$$$$$$$$$$$$$$         PREMIUMS        $$$$$$$$$$$$$$  \n")
  cat("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  \n")  
  cat("                                           \n")  
  cat("$$$ expected freq                          \n")
  cat("                                           \n") 
  print(summary(freq_split$f_final))
  cat("                                           \n")  
  cat("$$$ expected claims                        \n")
  cat("                                           \n") 
  print(summary(expected_claims))

  cat("                                           \n")  
  cat("$$$ Proportion of golden features: ", gf_proportion  ," \n")
  cat("                                           \n") 

  
 # Calculating various loading 
  
  premium_loading_fix = 0
  freq_loading = ((e_freq - 1)*0.63)+1
  premium_loading_variable = (1 - 0.13)
  
  expected_claims[expected_claims<0.99] <- 0.99
  
  predict_premium = (expected_claims + premium_loading_fix) * freq_loading / premium_loading_variable
  
  avg_loading = predict_premium / expected_claims
    

  cat("                                           \n")  
  cat("$$$ premium (with loading)                 \n")
  cat("                                           \n") 
  print(summary(predict_premium)) 
  cat("                                           \n")  
  cat("$$$ loading related to freq risk           \n")
  cat("                                           \n") 
  print(summary(freq_loading))
  cat("                                           \n")  
  cat("$$$ loading overall                        \n")
  cat("                                           \n") 
  print(summary(avg_loading)) 
  cat("                                           \n") 
  
  return(predict_premium) 
}



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SAVE & LOAD ####################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

save_model <- function(model,golden_feature=FALSE){
  
  # Two sets of models are fitted.
  # Regular, without the golden features
  # GF,      with    the golden features

  if (golden_feature == FALSE){
    
    save_model_basic(model)
    
  }
  else if (golden_feature == TRUE){
    
    save_model_gf(model)
    
  }

}

save_model_basic <- function(model){

  # lightgbm can NOT be saved into a RData file,
  # it needs to be saved on its own.
  
  # moreover, it needs to be saved FIRST for some weird reason I don't get.
  
  # extracting lgbs
  model_lgb_freq     <- model[[14]] 
  model_lgb_sev      <- model[[15]] 
  model_lgb_lcost    <- model[[16]]
  model_lgb_ens      <- model[[28]]
  
  # saving each into RDS file
  saveRDS.lgb.Booster(model_lgb_freq, file="model_lgb_freq.RDS")
  saveRDS.lgb.Booster(model_lgb_sev,  file="model_lgb_sev.RDS")
  saveRDS.lgb.Booster(model_lgb_lcost,file="model_lgb_lcost.RDS")
  saveRDS.lgb.Booster(model_lgb_ens,  file="model_lgb_ens.RDS")
  
  # removing lgb models from the list
  # can't just use NULL as it will collapse the list.
  # need to fill in dummy values.
  model[[14]] <- 123
  model[[15]] <- 123
  model[[16]] <- 123
  model[[28]] <- 123
  
  # destroying phantom values that bloat the models file size
  attr(model[[1]]$terms, ".Environment") <- NULL  
  attr(model[[2]]$terms, ".Environment") <- NULL  
  attr(model[[3]]$terms, ".Environment") <- NULL  
  attr(model[[4]]$terms, ".Environment") <- NULL  
  attr(model[[5]]$terms, ".Environment") <- NULL  
  attr(model[[6]]$terms, ".Environment") <- NULL  
  attr(model[[7]]$terms, ".Environment") <- NULL  
  #attr(model[[8]]$terms, ".Environment") <- NULL  
  #attr(model[[9]]$terms, ".Environment") <- NULL  
  #attr(model[[10]]$terms, ".Environment") <- NULL  

  attr(model[[25]]$terms, ".Environment") <- NULL    #the ens_xgb
  
  # for rf... 
  attr(model[[11]]$terms, ".Environment") <- NULL 
  attr(model[[12]]$terms, ".Environment") <- NULL 
  attr(model[[13]]$terms, ".Environment") <- NULL 
  attr(model[[27]]$terms, ".Environment") <- NULL 
  
  # for ann
  attr(model[[26]]$terms, ".Environment") <- NULL 
  attr(model[[26]]$finalModel$terms, ".Environment") <- NULL 
  

  # glm severity sometimes is bloated...   
  model[[6]]$dots = c()
  model[[6]]$finalModel$param$family$variance = c()
  model[[6]]$finalModel$param$family$dev.resids = c()
  model[[6]]$finalModel$param$family$aic = c()
  model[[6]]$finalModel$param$family$validmu = c()
  model[[6]]$finalModel$param$family$simulate = c()
  
  # removing training data that are hidden in the model objects
  model[[25]]$trainingData <- NULL
  model[[26]]$trainingData <- NULL
  model[[26]]$finalModel$fitted.values <- NULL  
  model[[26]]$finalModel$residuals <- NULL
  

  #finally saved the trimmed down model, fiou! done!
  save(model, file='trained_model.RData')
}




save_model_gf <- function(model){
  
  # lightgbm can NOT be saved into a RData file,
  # it needs to be saved on its own.
  
  # moreover, it needs to be saved FIRST for some weird reason I don't get.
  
  # extracting lgbs
  model_lgb_freq     <- model[[14]] 
  model_lgb_sev      <- model[[15]] 
  model_lgb_lcost    <- model[[16]]
  model_lgb_ens      <- model[[28]]
  
  # saving each into RDS file
  saveRDS.lgb.Booster(model_lgb_freq, file="model_lgb_freq_gf.RDS")
  saveRDS.lgb.Booster(model_lgb_sev,  file="model_lgb_sev_gf.RDS")
  saveRDS.lgb.Booster(model_lgb_lcost,file="model_lgb_lcost_gf.RDS")
  saveRDS.lgb.Booster(model_lgb_ens,  file="model_lgb_ens_gf.RDS")
  
  # removing lgb models from the list
  # can't just use NULL as it will collapse the list.
  # need to fill in dummy values.
  model[[14]] <- 123
  model[[15]] <- 123
  model[[16]] <- 123
  model[[28]] <- 123
  
  # destroying phantom values that bloat the models file size
  attr(model[[1]]$terms, ".Environment") <- NULL  
  attr(model[[2]]$terms, ".Environment") <- NULL  
  attr(model[[3]]$terms, ".Environment") <- NULL  
  attr(model[[4]]$terms, ".Environment") <- NULL  
  attr(model[[5]]$terms, ".Environment") <- NULL  
  attr(model[[6]]$terms, ".Environment") <- NULL  
  attr(model[[7]]$terms, ".Environment") <- NULL  
  #attr(model[[8]]$terms, ".Environment") <- NULL  
  #attr(model[[9]]$terms, ".Environment") <- NULL  
  #attr(model[[10]]$terms, ".Environment") <- NULL  
  
  attr(model[[25]]$terms, ".Environment") <- NULL    #the ens_xgb
  
  # for rf... 
  attr(model[[11]]$terms, ".Environment") <- NULL 
  attr(model[[12]]$terms, ".Environment") <- NULL 
  attr(model[[13]]$terms, ".Environment") <- NULL 
  attr(model[[27]]$terms, ".Environment") <- NULL 
  
  # for ann
  attr(model[[26]]$terms, ".Environment") <- NULL 
  attr(model[[26]]$finalModel$terms, ".Environment") <- NULL 
  
  
  # glm severity sometimes is bloated...   
  model[[6]]$dots = c()
  model[[6]]$finalModel$param$family$variance = c()
  model[[6]]$finalModel$param$family$dev.resids = c()
  model[[6]]$finalModel$param$family$aic = c()
  model[[6]]$finalModel$param$family$validmu = c()
  model[[6]]$finalModel$param$family$simulate = c()
  
  # removing training data that are hidden in the model objects
  model[[25]]$trainingData <- NULL
  model[[26]]$trainingData <- NULL
  model[[26]]$finalModel$fitted.values <- NULL  
  model[[26]]$finalModel$residuals <- NULL
  
  
  #finally saved the trimmed down model, fiou! done!
  save(model, file='trained_model_gf.RData')
}



load_model <- function(){ 
  cat(" ................................................. \n")
  cat(" .......       LOADING MODEL regular       ....... \n")
  cat(" ................................................. \n")
  model    <- load_model_basic()
  cat(" ................................................. \n")   
  cat(" .......       LOADING MODEL golden        ....... \n")
  cat(" ................................................. \n")  
  model_gf <- load_model_gf()
  
  model[[30]] <- model_gf
  
  return(model)
}



load_model_basic <- function(){ 
  
  #first load the lightgbm models
  model_lgb_freq  <- readRDS.lgb.Booster(file="model_lgb_freq.RDS")
  model_lgb_sev   <- readRDS.lgb.Booster(file="model_lgb_sev.RDS")
  model_lgb_lcost <- readRDS.lgb.Booster(file="model_lgb_lcost.RDS")
  model_lgb_ens   <- readRDS.lgb.Booster(file="model_lgb_ens.RDS")
  
  #then load all the other models
  load('trained_model.RData')
  
  #add back the lightgbm models to the list
  model[[14]] <- model_lgb_freq
  model[[15]] <- model_lgb_sev
  model[[16]] <- model_lgb_lcost
  model[[28]] <- model_lgb_ens
  
  
  #attr(model$terms, ".Environment") <- globalenv()
  return(model)
}



load_model_gf <- function(){ 
  
  #first load the lightgbm models
  model_lgb_freq  <- readRDS.lgb.Booster(file="model_lgb_freq_gf.RDS")
  model_lgb_sev   <- readRDS.lgb.Booster(file="model_lgb_sev_gf.RDS")
  model_lgb_lcost <- readRDS.lgb.Booster(file="model_lgb_lcost_gf.RDS")
  model_lgb_ens   <- readRDS.lgb.Booster(file="model_lgb_ens_gf.RDS")
  
  #then load all the other models
  load('trained_model_gf.RData')
  
  #add back the lightgbm models to the list
  model[[14]] <- model_lgb_freq
  model[[15]] <- model_lgb_sev
  model[[16]] <- model_lgb_lcost
  model[[28]] <- model_lgb_ens
  
  
  #attr(model$terms, ".Environment") <- globalenv()
  return(model)
}



