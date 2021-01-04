
# customed list
attr_list = ['Female', 
                    'AgeLess16', 'Age17-45', 'Age46-60', 'Ageover60', 
                    'Front', 'Side', 'Back', 
                    'a-Backpack', 'a-ShoulderBag', 
                    'hs-Hat', 
                    'hs-Glasses', 
                    'ub-ShortSleeve', 'ub-LongSleeve', 
                    'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Coat', 
                    'ub-Black', 'ub-Blue', 'ub-Brown', 'ub-Green', 'ub-Grey', 'ub-Orange', 'ub-Pink', 'ub-Purple', 'ub-Red', 'ub-White', 'ub-Yellow', 
                    'lb-LongTrousers', 'lb-Shorts', 'lb-ShortSkirt', 'lb-Dress', 
                    'lb-Black', 'lb-Blue', 'lb-Brown', 'lb-Green', 'lb-Grey', 'lb-Orange', 'lb-Pink', 'lb-Purple', 'lb-Red', 'lb-White', 'lb-Yellow', 
                    ]

# original pa100k list
org_pa100k_list = ['Female', 
    'AgeOver60', 'Age18-60', 'AgeLess18', 
    'Front', 'Side', 'Back', 'Hat', 'Glasses', 
    'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 
    'ShortSleeve', 'LongSleeve', 
    'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 
    'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots']

# reordered peta list
recodered_peta_list = ['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong'
    ,'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck'
    , 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers'
    , 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker'
    , 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags'
    , 'personalLess30','personalLess45','personalLess60','personalLarger60'
    , 'personalMale']

# reordered pa10k list
recodered_pa100k_list= ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo'
        , 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts'
        , 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60'
        , 'Age18-60', 'AgeLess18', 'Female', 'Front', 'Side', 'Back']

# customed 2 pa100k
pa100k_convert_dict={
    'Female': 'Female',
    # 'Male': None,
    'AgeLess16': 'AgeLess18',
    'Age17-45': None, # 'Age18-60',
    'Age46-60': None, # 'Age18-60',
    'Ageover60': 'AgeOver60',
    'Front': 'Front',
    'Side': 'Side',
    'Back': 'Back',
    'a-Backpack': 'Backpack',
    'a-ShoulderBag': 'ShoulderBag',
    'hs-Hat': 'Hat',
    'hs-Glasses': 'Glasses',
    'ub-ShortSleeve': 'ShortSleeve',
    'ub-LongSleeve': 'LongSleeve',
    'ub-Shirt': None,
    'ub-Sweater': None,
    'ub-Vest': None,
    'ub-TShirt': None,
    'ub-Cotton': None,
    'ub-Jacket': None,
    'ub-SuitUp': None,
    'ub-Coat': 'LongCoat', # None
    'ub-Black': None,
    'ub-Blue': None,
    'ub-Brown': None,
    'ub-Green': None,
    'ub-Grey': None,
    'ub-Orange': None,
    'ub-Pink': None,
    'ub-Purple': None,
    'ub-Red': None,
    'ub-White': None,
    'ub-Yellow': None,
    'lb-LongTrousers': 'Trousers',
    'lb-Shorts': 'Shorts',
    'lb-ShortSkirt': None, #'Skirt&Dress',
    'lb-Dress': None, #'Skirt&Dress',
    'lb-Black': None,
    'lb-Blue': None,
    'lb-Brown': None,
    'lb-Green': None,
    'lb-Grey': None,
    'lb-Orange': None,
    'lb-Pink': None,
    'lb-Purple': None,
    'lb-Red': None,
    'lb-White': None,
    'lb-Yellow': None
}

# customed 2 peta
convert_dict = {
    'Female': None,
    # 'Male': 'personalMale',
    'AgeLess16': 'personalLess15',
    'Age17-45': ['personalLess45', 'personalLess30'],
    'Age46-60': 'personalLess60',
    'Ageover60': 'personalLarger60',
    'Front': None,
    'Side': None,
    'Back': None,
    'a-Backpack': 'carryingBackpack',
    'a-ShoulderBag': 'carryingMessengerBag',
    'hs-Hat': 'accessoryHat',
    'hs-Glasses': 'accessorySunglasses',
    'ub-ShortSleeve': 'upperBodyShortSleeve',
    'ub-LongSleeve': 'upperBodyLongSleeve',
    'ub-Shirt': 'upperBodyTshirt',
    'ub-Sweater': 'upperBodySweater',
    'ub-Vest': None,
    'ub-TShirt': 'upperBodyTshirt',
    'ub-Cotton': None,
    'ub-Jacket': 'upperBodyJacket',
    'ub-SuitUp': 'upperBodySuit',
    'ub-Coat': None,
    'ub-Black': 'upperBodyBlack',
    'ub-Blue': 'upperBodyBlue',
    'ub-Brown': 'upperBodyBrown',
    'ub-Green': 'upperBodyGreen',
    'ub-Grey': 'upperBodyGrey',
    'ub-Orange': 'upperBodyOrange',
    'ub-Pink': 'upperBodyPink',
    'ub-Purple': 'upperBodyPurple',
    'ub-Red': 'upperBodyRed',
    'ub-White': 'upperBodyWhite',
    'ub-Yellow': 'upperBodyYellow',
    'lb-LongTrousers': 'lowerBodyTrousers',
    'lb-Shorts': 'lowerBodyShorts',
    'lb-ShortSkirt': 'lowerBodyShortSkirt',
    'lb-Dress': 'lowerBodyLongSkirt',
    'lb-Black': 'lowerBodyBlack',
    'lb-Blue': 'lowerBodyBlue',
    'lb-Brown': 'lowerBodyBrown',
    'lb-Green': 'lowerBodyGreen',
    'lb-Grey': 'lowerBodyGrey',
    'lb-Orange': 'lowerBodyOrange',
    'lb-Pink': 'lowerBodyPink',
    'lb-Purple': 'lowerBodyPurple',
    'lb-Red': 'lowerBodyRed',
    'lb-White': 'lowerBodyWhite',
    'lb-Yellow': 'lowerBodyYellow'
}

# customed 2 baidu
baidu_dict ={
    'Female': {'gender':'女性'},
    #'Male': {'gender':'男性'},
    'AgeLess16': {'age':['幼儿', '青少年']},
    'Age17-45': {'age':'青年'},
    'Age46-60': {'age':'中年'},
    'Ageover60': {'age':'老年'},
    'Front': {'orientation':'正面'},
    'Side': {'orientation':['左侧面', '右侧面']},
    'Back': {'orientation':'背面'},
    'a-Backpack': {'bag':'双肩包'},
    'a-ShoulderBag': {'bag':'单肩包'},
    'hs-Hat': {'headwear':'普通帽'},
    'hs-Glasses': {'glasses':'戴眼镜'},
    'ub-ShortSleeve': {'upper_wear':'短袖'},
    'ub-LongSleeve': {'upper_wear':'长袖'},
    'ub-Shirt': {'upper_wear_fg':'衬衫'},
    'ub-Sweater': {'upper_wear_fg':'毛衣'},
    'ub-Vest': {'upper_wear_fg':'无袖'},
    'ub-TShirt': {'upper_wear_fg':'T恤'},
    'ub-Cotton': {'upper_wear_fg':'羽绒服'},
    'ub-Jacket': {'upper_wear_fg':['外套', '夹克']}, #
    'ub-SuitUp': {'upper_wear_fg':'西装'},
    'ub-Coat': {'upper_wear_fg':'风衣'}, #coat这里仅指风衣
    'ub-Black': {'upper_color': '黑'},
    'ub-Blue': {'upper_color': '蓝'},
    'ub-Brown': {'upper_color': '棕'},
    'ub-Green': {'upper_color': '绿'},
    'ub-Grey': {'upper_color': '灰'},
    'ub-Orange': {'upper_color': '橙'},
    'ub-Pink': {'upper_color': '粉'},
    'ub-Purple': {'upper_color': '紫'},
    'ub-Red': {'upper_color': '红'},
    'ub-White': {'upper_color': '白'},
    'ub-Yellow': {'upper_color': '黄'},
    'lb-LongTrousers': {'lower_wear':'长裤'},
    'lb-Shorts': {'lower_wear':'短裤'},
    'lb-ShortSkirt': {'lower_wear':'短裙'},
    'lb-Dress': {'lower_wear':'长裙'},
    'lb-Black': {'lower_color': '黑'},
    'lb-Blue': {'lower_color': '蓝'},
    'lb-Brown': {'lower_color': '棕'},
    'lb-Green': {'lower_color': '绿'},
    'lb-Grey': {'lower_color': '灰'},
    'lb-Orange': {'lower_color': '橙'},
    'lb-Pink': {'lower_color': '粉'},
    'lb-Purple': {'lower_color': '紫'},
    'lb-Red': {'lower_color': '红'},
    'lb-White': {'lower_color': '白'},
    'lb-Yellow': {'lower_color': '黄'}
}
