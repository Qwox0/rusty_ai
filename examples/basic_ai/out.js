let training_x = '[-5.8130538399242395, 9.636433267725923, -2.9201887234089288, 7.70290805487349, 9.950807232916958, -7.393835462144249, -2.704407674383557, -9.060236112796414, 5.795703299568016, 6.475867495892267, -6.8964074198296235, 0.8498102512410028, 8.904184404191408, -6.468623111839076, -5.761128837790118, -2.8118321064373033, -4.016921217932076, -9.762036118910968, -0.03143020417618203, 6.132685201490762, -5.872888561478503, 4.506698878689688, 4.686407694917916, 4.313230102331753, 4.641929455442273, 9.919025527827017, -1.4527011208982579, 2.321963900179263, 5.187205250007796, 5.449881506287543, 7.861475270024631, -0.9853828418785717, 7.989862780079537, -9.957965622323215, -6.080451987302994, 5.282339295351395, -1.712031870403269, -2.042115600145163, 2.307600882304717, -2.1782673159604586, 5.823979804564704, 1.1020644356591518, 9.0775983802755, -1.1094250069348845, -5.200057424053005, -3.4338592470508313, 7.120415333566619, 6.522543356182204, 9.330242103210551, 2.2852277440130813, 6.215827123510909, 8.513704190976348, 9.017671981951871, -5.178036855321419, 9.64149090701688, -9.50971239783331, -3.539314588886916, 3.4730645112105947, -8.078650010451142, 5.43592273308048, 0.7952309320539523, -0.3118235221606973, -3.547159725454229, -9.41228878263308, 7.158760834661308, -9.858074532197868, -8.57897456041861, -5.0101464319457625, 4.726436285803402, -2.2830544560921977, 3.7822907287200263, -3.5400742131913834, 4.793993352231542, 9.957815815660506, -9.730902224762872, -1.289338573755579, -6.211000090620015, -8.52840497217092, -9.746706106174145, -6.587162156520993, 6.178083832777627, 7.328628814512509, 4.895108352500504, 3.598695372628036, 7.859146337333183, 7.519067613296649, -6.254053421447292, -2.0054219029424063, -6.476315003478428, -0.1964279652194314, 0.6314743985354347, -2.363760878172852, -0.0072830632878719825, 5.135124087910235, 6.801470663270887, 2.788616450931716, 4.370389153163686, 9.463021645341474, 2.3993908814505556, -2.3315760750588304]';
let training_y = '[0.4530034935007725, -0.21007855488934618, -0.21959949944589816, 0.9886100744960284, -0.5021033928554887, -0.8959875952458122, -0.4233908860406896, -0.3565212863347921, -0.4684026911463269, 0.19149213315569907, -0.5755054800222928, 0.7511551606333815, 0.4973951500683066, -0.18437684995601653, 0.4986637294952762, -0.32381648657407475, 0.7677540695134478, 0.3309009576487507, -0.031425029670056746, -0.14993260380634346, 0.39888146209160114, -0.9789202693543952, -0.9996625053882245, -0.9213882164596473, -0.9975187544485002, -0.4743693994170537, -0.9930348617487771, 0.7308925070515124, -0.8893767335781024, -0.7401569948981408, 0.9999719228407628, -0.833483688508363, 0.9907823528145551, 0.5082811050670998, 0.2013474207844935, -0.8419277857594064, -0.9900428288371781, -0.8909700316824015, 0.7406194344564376, -0.8210941831241276, -0.4432360555583961, 0.8921418802889665, 0.34024702201017226, -0.8954428603390727, 0.8834277501373959, 0.2881234380157096, 0.7427914111900984, 0.2370790261094577, 0.09439510886056485, 0.7554658919103828, -0.06730725980697586, 0.7901623060681267, 0.39595349535435237, 0.8935308561457407, -0.2150206225304197, 0.08483235638746642, 0.38731909712569457, -0.3254351165273852, -0.9748680407843955, -0.7494709375806434, 0.7140253040979377, -0.3067947296546796, 0.3945398941004118, -0.012488853462724542, 0.7679122864057955, 0.4198650023764451, -0.7485041126730446, 0.9559968047444342, -0.9999013382276369, -0.7568880188072501, -0.5977552189082435, 0.38801931769009396, -0.9966722105844511, -0.5081520871814568, 0.3013653609867359, -0.9606515590244971, 0.07212254357295626, -0.7810671757184943, 0.3163962386525572, -0.29931708906278726, -0.10490808382716452, 0.8651470500655247, -0.9833532076059919, -0.44135013145426827, 0.9999866629492545, 0.9444385734922328, 0.02912776536326247, -0.9070277648330936, -0.19193134004373044, -0.19516723888389834, 0.5903354717291214, -0.7017363470714195, -0.007282998902108571, -0.9119702576972856, 0.49539140728802006, 0.34569204823703986, -0.9420858642252232, -0.03823436284954093, 0.6759122154067589, -0.7242986051417272]';
let gen0_result = { gen: 0, error: '12841.402167708844', outputs: [2.724671886351267, 30.56443152562489, 2.7978975725400232, 25.670193295640285, 31.360191037175273, 2.6846583308139556, 2.8033595331724794, 2.6424775424969456, 20.842578758328916, 22.56424527200523, 2.6972494846565844, 8.323280495428756, 28.710925629786892, 2.708077780561069, 2.7259862386485505, 2.800640350827729, 2.7701365168936745, 2.6247132206412087, 6.092639437302045, 21.69556463937163, 2.72315731917746, 17.57978462802897, 18.034672804714255, 17.090066534225347, 17.922087202728257, 31.279743554498896, 3.2200042448692154, 12.049671349373545, 19.302317272607095, 19.967216890697404, 26.071566766881464, 3.85695954723671, 26.396547820030626, 2.6197537525173487, 2.7179033672400137, 19.54312544960875, 2.888840301402635, 2.8201238099846315, 12.01331494154794, 2.8166774678930375, 20.914153698673747, 8.961799230455522, 29.149879984104466, 3.6878899467162602, 2.7401883661647086, 2.7848952806758507, 24.195757831154094, 22.68239360455861, 29.78938473901285, 11.956682905914894, 21.906017738488497, 27.722522078146458, 28.998191209781165, 2.7407457620994276, 30.577233681710947, 2.631100168139082, 2.78222594095305, 14.963396280199232, 2.667323953581663, 19.93188372771635, 8.185126522180214, 5.382893483543553, 2.7820273608308366, 2.6335662046761117, 24.292819931072966, 2.6222822470457707, 2.6546594818510973, 2.744995490642255, 18.1359952278727, 2.814025042034639, 15.746125548681595, 2.7822067129528842, 18.306999140839753, 31.377931521433496, 2.625501297745014, 3.4426675672392903, 2.714598866663248, 2.6559395252166103, 2.6251012617895935, 2.7050772594384496, 21.810479984336464, 24.722798477777054, 18.56294661783757, 15.281399562870709, 26.065671652952886, 25.20484693687455, 2.7135090786622316, 2.82105261967615, 2.7078830794531035, 5.674988638340695, 7.770617582018154, 2.811982159667619, 6.153761919333736, 19.170486762766533, 23.38842871631501, 13.2308862302714, 17.234750456582155, 30.12548312911262, 12.245658497227298, 2.8127968379184027] };
var generations = [gen0_result];
let gen1_result = { gen: 1, error: '14467.213738167613', outputs: [-0.4600509093174022, -34.89292805773368, -0.005258881841187389, -28.194049773097827, -35.982105956565555, -0.9957612355695185, -0.005268152656593945, -1.5685530280230737, -21.58636116642997, -23.942853408918797, -0.8271880847372832, -4.450853794022173, -32.355983631938386, -0.6822164656484213, -0.44245407023292016, -0.005263537274083652, -0.00484310052580755, -1.8528984227230856, -1.39771411366191, -22.753866368896578, -0.4803282694248809, -17.12048522831235, -17.7431031691526, -16.450194623315646, -17.589004163573744, -35.871995277552685, -0.0053219309677707315, -9.551267300382063, -19.478162967954592, -20.38822933656736, -28.74342066682624, -0.028579950803478985, -29.188231159809927, -1.932282366148186, -0.550669339313008, -19.807763733564283, -0.00531078908335241, -0.005296607368410496, -9.501505286469824, -0.00529075774709152, -21.684327753071095, -5.324811923440282, -32.95679251248476, -0.014231795897861168, -0.2523128477649302, -0.005236812504579829, -26.1759494577859, -24.104566276604487, -33.832100234395675, -9.423991463911552, -23.041919305709325, -31.003128544505927, -32.749171920967186, -0.24485030778240646, -34.910450720377284, -1.7506654682675515, -0.005232281722047194, -13.539362638064823, -1.227837709599163, -20.33986786597706, -4.261758656466162, -0.42626533372463093, -0.005231944663657296, -1.71119274658637, -26.30880101973567, -1.8918099078962123, -1.3973924565215128, -0.1915363573293414, -17.881785952529277, -0.005286255673172007, -14.610705685336537, -0.005232249085554153, -18.11584370479491, -36.006387844608575, -1.8402840320947622, -0.005328949675047137, -0.5949107234716496, -1.3802549529993249, -1.8466872251106186, -0.7223881056824082, -22.911154157676492, -26.897324465882747, -18.466166046107137, -13.974622464952647, -28.735351862484087, -27.557117431212223, -0.6095010461193345, -0.005298183876147794, -0.6848231671191952, -0.8260639944293995, -3.6944088856627566, -0.0052827882030460965, -1.4813741335352004, -19.297722928437725, -25.07093592517599, -11.168028549693027, -16.64822748155986, -34.29212733179905, -9.819520292396582, -0.00528417099069086] };
var generations = [gen0_result,gen1_result];
let gen2_result = { gen: 2, error: '17637333865438.004', outputs: [853512.2064659768, -150.83851800304805, 403841.7145896485, -120.89412918312127, -155.70720871675994, 1099230.8352242315, 370300.44441466354, 1358258.1904797102, -91.38193277010394, -101.89102575746233, 1021910.0119394822, -24.266534468822947, -139.49822384144275, 955414.6958969561, 845440.9206265404, 386998.6295202765, 574319.1549146356, 1467346.8417779366, -14.379593513964892, -96.57618217985365, 862812.9887661895, -72.28638046880165, -74.94862044541314, -69.42029787220781, -74.28971149644444, -155.21500735923567, 175733.6468210851, -41.183381613550054, -82.36752898497183, -86.25886411839387, -123.34985011865226, 103093.12196913103, -125.3381797743255, 1497802.363423065, 895076.9015401335, -83.77686250553128, 216044.33570278643, 267352.9543680787, -41.0088626903786, 288516.5435969532, -91.8008261977109, -27.096661265234516, -142.18387567166167, 122374.38781724844, 758227.2938478307, 483687.28903753776, -111.87309939998319, -102.61389200184118, -146.09655383183957, -40.737016199848085, -97.86379613164401, -133.45088023646852, -141.2557991407468, 754804.3897115822, -150.9168453589225, 1428125.333965042, 500079.39635139, -56.97392784651261, 1205679.2461864976, -86.05207624434155, -23.65419024104546, 4281.7099719955295, 501298.8539745337, 1412981.6880627992, -112.46695387192999, 1482275.1702005777, 1283450.3061634256, 728707.2969609888, -75.54161146888877, 304804.7848206271, -61.554862802004585, 500197.4732823472, -76.54241447189327, -155.81575021668684, 1462507.3511927465, 150340.37269003747, 915369.4583886876, 1275589.7075162565, 1464963.9258485062, 973840.5499833976, -97.27926805538812, -115.09768907150666, -78.04035097607256, -58.83504689777214, -123.3137820779644, -118.04700330815906, 922061.720789548, 261649.2411960659, 956610.3329410688, -12.52842666036634, -21.81694914763316, 317349.889778131, -14.650508625412353, -81.59598891991223, -106.93362245088811, -47.07368083755315, -70.26706272962407, -148.15290262443415, -42.12416394174232, 312347.04461794655] };
var generations = [gen0_result,gen1_result,gen2_result];
