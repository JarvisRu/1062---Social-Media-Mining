import xml.etree.ElementTree as ET

# get all aspectTerms from train
def getTermsFromTrain():
    global aspectTerms
    tree = ET.parse('Restaurants_Train.xml')
    root = tree.getroot()
    
    for sentence in root.findall('sentence'):
        aspeTerms = sentence.find('aspectTerms')
        if aspeTerms is not None:
            for aspeTerm in aspeTerms.findall('aspectTerm'):
                if aspeTerm.get('term') not in aspectTerms:
                    aspectTerms.append(aspeTerm.get('term'))
    # print(aspectTerms)

# do some work on test data
def modifyTestData():
    global aspectTerms
    tree = ET.parse('test.xml')
    root = tree.getroot()
    print("in")
    for sentence in root.findall('sentence'):
        print("=======================================")
        txt = sentence.find('text').text
        print(txt)
        for aspeTerm in aspectTerms:
            if aspeTerm in txt:
                print("txt include ", aspeTerm)


def main():
    global aspectTerms
    aspectTerms = []
    getTermsFromTrain()
    modifyTestData()

main()