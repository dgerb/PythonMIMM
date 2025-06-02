"""
EnergyModel.py
    v1.01, 3/17/20
    v1.02, 6/2/25

zones/portsVIMult: [[port voltage, port current, 1 or -1 for direction (positive power flows into zone)],[...]]
subcircuits/portVIMult: [port voltage, port current, 1 or -1 for direction (positive power flows into branch)]
subcircuits/subcircuitType: "load", "grid", "source", "battery"
subcircuits/branches/converterMaxPowerXXType: "percentage", "oversizeRatio", "fixed"

converterMaxPowerXXType details
fixed: converterMaxPowerXX is the actual converter's max rated power of converter (W)
percentage: assume all loads operate at a fixed percentage of rated max power on the efficiency curve. converterMaxPowerXX is the fixed percentage (0-1)
oversizeRatio: assume the converter's max rated power is the peak metered power multiplied by an oversize ratio. converterMaxPowerXX as the oversize ratio (>1)
"""
import sys, os, time, numpy as np, pandas as pd, csv
import packageDC.DCBuildingsPython as dc
import json, copy

PROJECTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PROJECTPATH)

effData = dc.getEffData(PROJECTPATH, "")[1]

def effLerp(effCurve, percentPower):
    x = effCurve[:,0]
    y = effCurve[:,1]
    percentPower = np.clip(percentPower, 0.01, 0.99)
    for n in range(0, len(x)):
        if x[n] > percentPower:
            break
    x1 = x[n-1]
    x3 = x[n]
    y1 = y[n-1]
    y3 = y[n]
    y2 = (percentPower - x1)*(y3 - y1)/(x3-x1) + y1
    return y2

# wire gauge resistance chart in ohms/km 
GAUGERESDC = {'23':66.625, '18':26.1,'16':16.4,'14':10.3,'12':6.5,'10':4.07,'8':2.551, \
    '6':1.608,'4':1.01,'3':0.802,'2':0.634,'1':0.505,'1/0':0.399, \
    '2/0':0.317,'3/0':0.2512,'4/0':0.1996,'250':0.1687,'300':0.1409, \
    '350':0.1205,'400':0.1053,'500':0.0845,'600':0.0704}
GAUGERESAC = {'14':10.2,'12':6.6,'10':3.9,'8':2.56,'6':1.61,'4':1.02, \
    '3':0.82,'2':0.66,'1':0.52,'1/0':0.43,'2/0':0.33,'3/0':0.269, \
    '4/0':0.220,'250':0.187,'300':0.161,'350':0.141,'400':0.125}

# subcircuit branch templates
BRANCHTEMPLATES = {\
    "loadOnly": [ {
        "name": "loadOnly",
        "converterTagDC": "1",
        "converterMaxPowerDC": 1.0,
        "converterMaxPowerDCType": "oversizeRatio",
        "converterTagAC": "1",
        "converterMaxPowerAC": 1.0,
        "converterMaxPowerACType": "oversizeRatio"
                } ],
    "light48": [ {
        "name": "light",
        "converterTagDC": "DC-DC LED Driver LL",
        "converterMaxPowerDC": 0.75,
        "converterMaxPowerDCType": "percentage",
        "converterTagAC": "Rectifier LED LL",
        "converterMaxPowerAC": 0.75,
        "converterMaxPowerACType": "percentage"
                } ],
    "fan48": [ {
        "name": "fan",
        "converterTagDC": "1",
        "converterMaxPowerDC": 1.25,
        "converterMaxPowerDCType": "oversizeRatio",
        "converterTagAC": "Rectifier LL 0-1",
        "converterMaxPowerAC": 1.25,
        "converterMaxPowerACType": "oversizeRatio"
                } ],
    "electronics48": [ {
        "name": "plug",
        "converterTagDC": "1",
        "converterMaxPowerDC": 1.25,
        "converterMaxPowerDCType": "oversizeRatio",
        "converterTagAC": "Rectifier LL 0-1",
        "converterMaxPowerAC": 1.25,
        "converterMaxPowerACType": "oversizeRatio"        
                } ],
    "light380": [ {
        "name": "light",
        "converterTagDC": "DC-DC LED Driver HL",
        "converterMaxPowerDC": 0.75,
        "converterMaxPowerDCType": "percentage",
        "converterTagAC": "Rectifier LED HL",
        "converterMaxPowerAC": 0.75,
        "converterMaxPowerACType": "percentage"
                } ],
    "fan380": [ {
        "name": "fan",
        "converterTagDC": "1",
        "converterMaxPowerDC": 1.25,
        "converterMaxPowerDCType": "oversizeRatio",
        "converterTagAC": "Rectifier HH",
        "converterMaxPowerAC": 1.25,
        "converterMaxPowerACType": "oversizeRatio"
                } ]
    }

json_file = open('Nodes.json')
nodeConfig = json.load(json_file)
json_file.close()

totalStatsCSVPower = []
totalStatsCSVPercent = []

for fileName in os.listdir(os.path.join(PROJECTPATH, 'dataCollection')):
    if not fileName[-4:] == '.csv':
        continue
    dataPath = os.path.join('dataCollection', fileName)
    print(fileName)
    data = np.genfromtxt(dataPath, dtype=float, delimiter=',', names=True, deletechars='')
    fileName = fileName[0:-4]
    totalStatsLinePower = [fileName]
    totalStatsLinePercent = [fileName]
    resultLength = len(data["time"])
    lossMatrixDC = data["time"].reshape((resultLength,1))
    lossMatrixAC = data["time"].reshape((resultLength,1))
    lossHeader = ["Time"]
    loadMatrix = data["time"].reshape((resultLength,1))
    loadHeader = ["Time"]
    # Zone loss (DC only)
    numZones = 0
    for zoneConfig in nodeConfig["zones"]:
        numZones = numZones + 1
        portsVIMult = zoneConfig["portsVIMult"]
        zoneTotalPower = np.zeros((resultLength, 1)) # total power going into zone
        for powerConfig in portsVIMult:
            vArr = data[powerConfig[0]].reshape((resultLength,1))
            iArr = data[powerConfig[1]].reshape((resultLength,1))
            sgn = int(powerConfig[2])
            zoneTotalPower = zoneTotalPower + np.multiply(vArr,iArr) * sgn
        zoneLossPower = np.clip(zoneTotalPower, 0, 1000000000)
        standbyDC = zoneConfig['assumedZoneStandbyDC']
        zoneLossStandbyPower = np.full(zoneLossPower.shape, standbyDC)
        zoneLossStandbyPower[zoneLossPower < standbyDC] = zoneLossPower[zoneLossPower < standbyDC]
        zoneLossConversionPower = zoneLossPower - zoneLossStandbyPower
        lossHeader.extend([zoneConfig["name"] + '/converter', zoneConfig["name"] + '/standby'])
        lossMatrixDC = np.hstack((lossMatrixDC, zoneLossConversionPower, zoneLossStandbyPower))
        lossMatrixAC = np.hstack((lossMatrixAC, np.zeros((resultLength,1)), \
            np.ones((resultLength,1))*zoneConfig['assumedZoneStandbyAC']))

    # Subcircuit loss
    numSubCircuits = 0
    for subConfig in nodeConfig["subcircuits"]:
        numSubCircuits = numSubCircuits + 1
        subName = subConfig["name"]
        subType = subConfig.get("subcircuitType", "load")
        branchTemplate = subConfig.get("branchTemplate", "none")
        branchTemplateOverwrite = subConfig.get("branchTemplateOverwrite", None)
        # get subcircuit power
        powerConfig = subConfig["portVIMult"]
        isPowerReading = powerConfig[0] == "Ones"
        vArr = data[powerConfig[0]].reshape((resultLength,1))
        iArr = data[powerConfig[1]].reshape((resultLength,1))
        sgn = int(powerConfig[2])
        subCircuitPower = np.multiply(vArr,iArr) * sgn
        # subcircuit branches
        if branchTemplate in BRANCHTEMPLATES:
            tempBranches = copy.deepcopy(BRANCHTEMPLATES[branchTemplate]) # tempBranches = array of dicts
            if branchTemplateOverwrite: # branchTemplateOverwrite = array of dicts
                for branchInfo in tempBranches: # branchInfo = dict
                    for overwriteInfo in branchTemplateOverwrite: # overwriteInfo = dict
                        if branchInfo["name"] == overwriteInfo["name"]:
                            for owkey in overwriteInfo.keys(): #overwriteItem = member of dict
                                branchInfo[owkey] = overwriteInfo[owkey]
            subConfig["branches"] = tempBranches
        for branchConfig in subConfig["branches"]:
            # parse dictionary and assign default values
            branchName = branchConfig["name"]
            powerFraction = branchConfig.get("powerFraction", 1.0) # fraction (0-1) of total subcircuit power we assume this branch draws
            branchPowerDC = subCircuitPower * powerFraction
            branchCurrentDC = iArr * powerFraction
            assumedVoltageDC = branchConfig.get("assumedVoltageDC", 48) # assumed branch voltage for when branchCurrentDC is a power reading
            assumedVoltageAC = branchConfig.get("assumedVoltageAC", 120) # assumed branch voltage for equivalent AC circuit
            effCurveDC = effData[branchConfig.get("converterTagDC", "1")] # converter efficiency curve "Part Type" column of packageDC/Converters.csv
            effCurveAC = effData[branchConfig.get("converterTagAC", "1")]
            maxPowerDC = branchConfig.get("converterMaxPowerDC", 0.75) # max power quantity, depends on max power type
            maxPowerAC = branchConfig.get("converterMaxPowerAC", 0.75)
            maxPowerDCType = branchConfig.get("converterMaxPowerDCType", "percentage") # see details at the top
            maxPowerACType = branchConfig.get("converterMaxPowerACType", "percentage")
            numberUnitsDC = branchConfig.get("numberUnitsDC", 1) # number of converter units on the branch
            numberUnitsAC = branchConfig.get("numberUnitsAC", 1)
            wireLengthDC = branchConfig.get("wireLengthDC", 0) # wire length (m) of subcircuit branch between meter and converter
            wireLengthAC = branchConfig.get("wireLengthAC", 0)
            wireGaugeDC = branchConfig.get("wireGaugeDC", "12") # wire gauge of subcircuit branch between meter and converter
            wireGaugeAC = branchConfig.get("wireGaugeAC", "12")
            branchStandbyDC = branchConfig.get("assumedBranchStandbyDC", 0) # assumed standby consumption in converter (for accounting)
            branchStandbyAC = branchConfig.get("assumedBranchStandbyAC", 0) # assumed standby consumption in converter (for accounting)
            # initialize
            branchPowerAC = np.zeros((resultLength, 1))
            standbyLossDC = np.zeros((resultLength, 1))
            standbyLossAC = np.zeros((resultLength, 1))
            converterLossDC = np.zeros((resultLength, 1))
            converterLossAC = np.zeros((resultLength, 1))
            wireLossDC = np.zeros((resultLength, 1))
            wireLossAC = np.zeros((resultLength, 1))
            loadProfile = np.zeros((resultLength, 1))
            # DC
            # get percent power DC array, referenced to input
            if maxPowerDCType == "oversizeRatio":
                maxPowerDC = np.max((np.abs(maxPowerDC*np.max(np.abs(branchPowerDC))), 0.001))
                percentPower = np.abs(np.divide(branchPowerDC, maxPowerDC))
            elif maxPowerDCType == "fixed": # fixed max power (W)
                percentPower = np.abs(np.divide(branchPowerDC, maxPowerDC * numberUnitsDC))
            else: # percentage: fixed percentage (%)
                percentPower = np.ones((resultLength, 1)) * np.clip(np.abs(maxPowerDC), 0, 1)
            # calculate wire loss DC
            gaugeResDict = GAUGERESDC
            wireR = wireLengthDC * (gaugeResDict[wireGaugeDC] / 1000) / numberUnitsDC
            wireR = wireR * 2 # must account for both positive and negative
            if isPowerReading: # branchCurrent is actually a power reading, divide by the assumed branch voltage
                wireLossDC = wireR * np.multiply(branchCurrentDC/assumedVoltageDC, branchCurrentDC/assumedVoltageDC)
            else:
                wireLossDC = wireR * np.multiply(branchCurrentDC, branchCurrentDC)
            # calculate converter DC loss and load
            # note that standby is subtracted from input side of power converters
            # note that we do not account for second order effects of wire voltage drop before input of a converter
            for n in range(0, resultLength):
                standbyLossDC[n] = min(np.abs(branchStandbyDC), np.abs(branchPowerDC[n]))
                if branchPowerDC[n] > 0:
                    loadProfile[n] = max((branchPowerDC[n] - standbyLossDC[n]), 0) * effLerp(effCurveDC, percentPower[n])
                else:
                    loadProfile[n] = (branchPowerDC[n] + standbyLossDC[n]) / effLerp(effCurveDC, percentPower[n])
                converterLossDC[n] = max(np.abs(branchPowerDC[n] - loadProfile[n]) - standbyLossDC[n], 0)
            # AC
            # get percent power AC array, referenced to input of DC system branchPowerDC
            if maxPowerACType == "oversizeRatio":
                maxPowerAC = np.max((np.abs(maxPowerAC*np.max(np.abs(branchPowerDC))), 0.001))
                percentPower = np.abs(np.divide(branchPowerDC, maxPowerAC))
            elif maxPowerDCType == "fixed":
                percentPower = np.abs(np.divide(branchPowerDC, maxPowerAC * numberUnitsAC))
            else: # percentage
                percentPower = np.ones((resultLength, 1)) * np.clip(np.abs(maxPowerAC), 0, 1)
            # calculate converter AC loss and load
            for n in range(0, resultLength):
                standbyLossAC[n] = branchStandbyAC
                if loadProfile[n] > 0:
                    branchPowerAC[n] = (loadProfile[n] / effLerp(effCurveAC, percentPower[n])) + standbyLossAC[n]
                else:
                    branchPowerAC[n] = min(loadProfile[n], 0) * effLerp(effCurveAC, percentPower[n]) - standbyLossAC[n]
                    # print(percentPower[n])
                    # print(branchPowerAC[n])
                    # print(branchPowerAC[n])
                    # print(loadProfile[n])
                converterLossAC[n] = max(np.abs(branchPowerAC[n] - loadProfile[n]) - standbyLossAC[n], 0)
            # calculate wire loss AC
            branchCurrentAC = branchPowerAC/assumedVoltageAC;
            gaugeResDict = GAUGERESAC
            wireR = wireLengthAC * (gaugeResDict[wireGaugeAC] / 1000) / numberUnitsAC
            wireR = wireR * 2 # must account for both hot and neutral
            wireLossAC = wireR * np.multiply(branchCurrentAC, branchCurrentAC)
            # output matrices
            # store to loss and load matrix
            lossHeader.extend([subName + "/" + branchName + "/" + "converter"])
            lossMatrixDC = np.hstack((lossMatrixDC, converterLossDC))
            lossMatrixAC = np.hstack((lossMatrixAC, converterLossAC))
            lossHeader.extend([subName + "/" + branchName + "/" + "standby"])
            lossMatrixDC = np.hstack((lossMatrixDC, standbyLossDC))
            lossMatrixAC = np.hstack((lossMatrixAC, standbyLossAC))
            lossHeader.extend([subName + "/" + branchName + "/" + "wire"])
            lossMatrixDC = np.hstack((lossMatrixDC, wireLossDC))
            lossMatrixAC = np.hstack((lossMatrixAC, wireLossAC))
            if subType == "load":
                loadHeader.extend([subName + "/" + branchName])
                loadMatrix = np.hstack((loadMatrix, loadProfile))

    # Total loss and load to find energy and efficiency
    zoneStartInd = 1
    branchStartInd = zoneStartInd+numZones*2
    totalLossDCP = np.sum(lossMatrixDC[:,1:], axis=1).reshape((resultLength,1))
    totalLossDCPZoneConv = np.sum(lossMatrixDC[:,zoneStartInd:branchStartInd:2], axis=1).reshape((resultLength,1))
    totalLossDCPZoneStby = np.sum(lossMatrixDC[:,zoneStartInd+1:branchStartInd:2], axis=1).reshape((resultLength,1))
    totalLossDCPBranchConv = np.sum(lossMatrixDC[:,branchStartInd::3], axis=1).reshape((resultLength,1))
    totalLossDCPBranchStby = np.sum(lossMatrixDC[:,branchStartInd+1::3], axis=1).reshape((resultLength,1))
    totalLossDCPBranchWire = np.sum(lossMatrixDC[:,branchStartInd+2::3], axis=1).reshape((resultLength,1))
    totalLossACP = np.sum(lossMatrixAC[:,1:], axis=1).reshape((resultLength,1))
    totalLossACPZoneConv = np.sum(lossMatrixAC[:,zoneStartInd:branchStartInd:2], axis=1).reshape((resultLength,1))
    totalLossACPZoneStby = np.sum(lossMatrixAC[:,zoneStartInd+1:branchStartInd:2], axis=1).reshape((resultLength,1))
    totalLossACPBranchConv = np.sum(lossMatrixAC[:,branchStartInd::3], axis=1).reshape((resultLength,1))
    totalLossACPBranchStby = np.sum(lossMatrixAC[:,branchStartInd+1::3], axis=1).reshape((resultLength,1))
    totalLossACPBranchWire = np.sum(lossMatrixAC[:,branchStartInd+2::3], axis=1).reshape((resultLength,1))
    totalLoadP = np.sum(loadMatrix[:,1:], axis=1).reshape((resultLength,1))
    timeStep = nodeConfig["timeStepSec"]
    totalLossDCEnergy = np.sum(totalLossDCP) * timeStep / 60.0 / 60.0
    totalLossDCEZoneConv = np.sum(totalLossDCPZoneConv) * timeStep / 60.0 / 60.0
    totalLossDCEZoneStby = np.sum(totalLossDCPZoneStby) * timeStep / 60.0 / 60.0
    totalLossDCEBranchConv = np.sum(totalLossDCPBranchConv) * timeStep / 60.0 / 60.0
    totalLossDCEBranchStby = np.sum(totalLossDCPBranchStby) * timeStep / 60.0 / 60.0
    totalLossDCEBranchWire = np.sum(totalLossDCPBranchWire) * timeStep / 60.0 / 60.0
    totalLossACEnergy = np.sum(totalLossACP) * timeStep / 60.0 / 60.0
    totalLossACEZoneConv = np.sum(totalLossACPZoneConv) * timeStep / 60.0 / 60.0
    totalLossACEZoneStby = np.sum(totalLossACPZoneStby) * timeStep / 60.0 / 60.0
    totalLossACEBranchConv = np.sum(totalLossACPBranchConv) * timeStep / 60.0 / 60.0
    totalLossACEBranchStby = np.sum(totalLossACPBranchStby) * timeStep / 60.0 / 60.0
    totalLossACEBranchWire = np.sum(totalLossACPBranchWire) * timeStep / 60.0 / 60.0
    totalLoadEnergy = np.sum(totalLoadP) * timeStep / 60.0 / 60.0
    efficiencyDC = 100*totalLoadEnergy/(totalLoadEnergy + totalLossDCEnergy)
    efficiencyAC = 100*totalLoadEnergy/(totalLoadEnergy + totalLossACEnergy)
    energyUnit = nodeConfig.get("energyUnit", "kWh")
    print("Loss DC: " + str(np.round(totalLossDCEnergy)) + " " + energyUnit + ", Load: " + \
        str(np.round(totalLoadEnergy)) + energyUnit + ", Efficiency:" + str(np.round(efficiencyDC, 1)))
    print("Loss AC: " + str(np.round(totalLossACEnergy)) + " " + energyUnit + ", Load: " + \
        str(np.round(totalLoadEnergy)) + energyUnit + ", Efficiency:" + str(np.round(efficiencyAC, 1)))
    # Total stats CSV data
    totalStatsHeader = ["Experiment", "Total Loss DC (" + energyUnit + ")", \
        "Total Loss AC (" + energyUnit + ")", \
        "Total Load (" + energyUnit + ")", "Efficiency DC", "Efficiency AC"]
    totalStatsHeader.extend(["DC Loss (" + energyUnit + ") ------------------"])
    totalStatsHeader.extend(["Zone Converter", "Zone Standby", "Branch Converter", "Branch Standby", "Branch Wire"])
    totalStatsHeader.extend(lossHeader[1:])
    totalStatsHeader.extend(["AC Loss (" + energyUnit + ") ------------------"])
    totalStatsHeader.extend(["Zone Converter", "Zone Standby", "Branch Converter", "Branch Standby", "Branch Wire"])
    totalStatsHeader.extend(lossHeader[1:])
    totalStatsHeader.extend(["Load (" + energyUnit + ") ------------------"])
    totalStatsHeader.extend(loadHeader[1:])
    lossDCEArr = np.sum(lossMatrixDC[:,1:], axis=0) * timeStep / 60.0 / 60.0
    lossACEArr = np.sum(lossMatrixAC[:,1:], axis=0) * timeStep / 60.0 / 60.0
    loadEArr = np.sum(loadMatrix[:,1:], axis=0) * timeStep / 60.0 / 60.0
    totalStatsLinePower.extend([totalLossDCEnergy, totalLossACEnergy, totalLoadEnergy, efficiencyDC, efficiencyAC])
    totalStatsLinePower.extend(["------------------------------------"])
    totalStatsLinePower.extend([totalLossDCEZoneConv, totalLossDCEZoneStby])
    totalStatsLinePower.extend([totalLossDCEBranchConv, totalLossDCEBranchStby, totalLossDCEBranchWire])
    totalStatsLinePower.extend(lossDCEArr)
    totalStatsLinePower.extend(["------------------------------------"])
    totalStatsLinePower.extend([totalLossACEZoneConv, totalLossACEZoneStby])
    totalStatsLinePower.extend([totalLossACEBranchConv, totalLossACEBranchStby, totalLossACEBranchWire])
    totalStatsLinePower.extend(lossACEArr)
    totalStatsLinePower.extend(["------------------------------------"])
    totalStatsLinePower.extend(loadEArr)
    totalStatsLinePercent.extend([totalLossDCEnergy, totalLossACEnergy, totalLoadEnergy, efficiencyDC, efficiencyAC])
    totalStatsLinePercent.extend(["------------------------------------"])
    totalStatsLinePercent.extend([np.round(100*totalLossDCEZoneConv/totalLossDCEnergy, 2), \
        np.round(100*totalLossDCEZoneStby/totalLossDCEnergy, 2)])
    totalStatsLinePercent.extend([np.round(100*totalLossDCEBranchConv/totalLossDCEnergy, 2), \
        np.round(100*totalLossDCEBranchStby/totalLossDCEnergy, 2), np.round(100*totalLossDCEBranchWire/totalLossDCEnergy, 2)])
    totalStatsLinePercent.extend(np.round(100*lossDCEArr/totalLossDCEnergy, 2))
    totalStatsLinePercent.extend(["------------------------------------"])
    totalStatsLinePercent.extend([np.round(100*totalLossACEZoneConv/totalLossACEnergy, 2), \
        np.round(100*totalLossACEZoneStby/totalLossACEnergy, 2)])
    totalStatsLinePercent.extend([np.round(100*totalLossACEBranchConv/totalLossACEnergy, 2), \
        np.round(100*totalLossACEBranchStby/totalLossACEnergy, 2), np.round(100*totalLossACEBranchWire/totalLossACEnergy, 2)])
    totalStatsLinePercent.extend(np.round(100*lossACEArr/totalLossACEnergy, 2))
    totalStatsLinePercent.extend(["------------------------------------"])
    totalStatsLinePercent.extend(np.round(100*loadEArr/totalLoadEnergy, 2))
    # Loss and load CSV data
    lossHeader.extend(["Total Loss"])
    lossMatrixDC = np.hstack((lossMatrixDC, totalLossDCP))
    lossMatrixAC = np.hstack((lossMatrixAC, totalLossACP))
    lossHeader.extend(["Total Load"])
    lossMatrixDC = np.hstack((lossMatrixDC, totalLoadP))
    lossMatrixAC = np.hstack((lossMatrixAC, totalLoadP))
    lossHeader.extend(["Efficiency"])
    lossMatrixDC = np.hstack((lossMatrixDC, np.divide(totalLoadP, totalLoadP + totalLossDCP + 0.01*np.ones_like(totalLoadP))))
    lossMatrixAC = np.hstack((lossMatrixAC, np.divide(totalLoadP, totalLoadP + totalLossACP + 0.01*np.ones_like(totalLoadP))))
    loadHeader.extend(["Total Load"])
    loadMatrix = np.hstack((loadMatrix, totalLoadP))
    # Write CSV of hourly loss, hourly load
    csvData = [lossHeader.copy()]
    csvData.extend(lossMatrixDC.tolist())
    dc.writeCSVFile(os.path.join('results', "LossDC " + fileName + '.csv'), csvData)
    csvData = [lossHeader.copy()]
    csvData.extend(lossMatrixAC.tolist())
    dc.writeCSVFile(os.path.join('results', "LossAC " + fileName + '.csv'), csvData)
    csvData = [loadHeader.copy()]
    csvData.extend(loadMatrix.tolist())
    dc.writeCSVFile(os.path.join('results', "Load " + fileName + '.csv'), csvData)
    totalStatsCSVPower.append(totalStatsLinePower)
    totalStatsCSVPercent.append(totalStatsLinePercent)
# Write CSV of total stats
csvData = []
csvData.append(totalStatsHeader)
csvData.extend(totalStatsCSVPower)
csvData = np.array(csvData).T.tolist()
dc.writeCSVFile(os.path.join('results', 'TotalStatsPower.csv'), csvData)
csvData = []
csvData.append(totalStatsHeader)
csvData.extend(totalStatsCSVPercent)
csvData = np.array(csvData).T.tolist()
dc.writeCSVFile(os.path.join('results', 'TotalStatsPercent.csv'), csvData)

