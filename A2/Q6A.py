# KARAN KUNWAR
# IIT2019001

import numpy as np

def findW(FloorArea, NoOfBedrooms, NoOfBathrooms, Price):

      FloorAreaTrain = FloorArea[:382]
      NoOfBathroomsTrain = NoOfBathrooms[:382]
      NoOfBedroomsTrain = NoOfBedrooms[:382]
      PriceTrain = Price[:382]

      Col1 = []
      for i in range(len(FloorAreaTrain)):
            Col1.append(1)

      table = zip(Col1, FloorAreaTrain, NoOfBedroomsTrain, NoOfBathroomsTrain)
      Y = []
      for i in PriceTrain:
            temp = []
            temp.append(i)
            Y.append(temp)

      # Using W = (X'X)^-1 X'Y
      X = list(table)
      XT = np.transpose(X)
      XTX = np.dot(XT, X)
      XTX_INV = np.linalg.inv(XTX)
      XTX_INV_XT = np.dot(XTX_INV, XT)

      W = np.dot(XTX_INV_XT, Y)
      Y_pred = []
      for i in range(383, 546):
            Y_pred.append(int(W[0][0] + W[1][0] * FloorArea[i] + W[2][0] * NoOfBedrooms[i] + W[3][0] * NoOfBathrooms[i]))
      error =0
      for i in range(163):
            error = error + abs((Y_pred[i] - Price[i + 382]) / Price[i + 382])
      error = error / 163

      return error, W

def findWAfterNormalisation(FloorArea, NoOfBedrooms, NoOfBathrooms, Price):
      Col1 = []
      for i in range(len(FloorArea)):
            Col1.append(1)

      FloorAreaTrain = FloorArea[:382]
      NoOfBathroomsTrain = NoOfBathrooms[:382]
      NoOfBedroomsTrain = NoOfBedrooms[:382]
      PriceTrain = Price[:382]

      table = zip(Col1, FloorAreaTrain, NoOfBedroomsTrain, NoOfBathroomsTrain)

      Y = []
      for i in PriceTrain:
            temp = []
            temp.append(i)
            Y.append(temp)

      # Using W = (X'X)^-1 X'Y
      X = list(table)
      XT = np.transpose(X)
      XTX = np.dot(XT, X)

      constant = np.identity(4, dtype=float)
      MatrixTemp = np.dot(XT, Y)
      constant[0][0] = 0
      error = 100
      WF = []
      for i in range(30):
            XTX_BAR = np.add(XTX, constant * i)
            XTX_BAR_INV = np.linalg.inv(XTX_BAR)
            W = np.dot(XTX_BAR_INV, MatrixTemp)
            Y_pred = []
            for i in range(383, 546):
                  Y_pred.append(int(W[0][0] + W[1][0] * FloorArea[i] + W[2][0] * NoOfBedrooms[i] + W[3][0] * NoOfBathrooms[i]))
            errort = 0
            for i in range(163):
                  errort = errort + abs((Y_pred[i] - Price[i + 382]) / Price[i + 382])
            errort = error / 163
            if errort < error:
                  error = errort
                  WF = W 
      return error, WF


def main():
      wines = np.genfromtxt("dataset.csv", delimiter=",", skip_header=1)
      
      Price=[]
      NoOfBathrooms=[]
      NoOfBedrooms=[]
      FloorArea=[]

      for i in range(len(wines)):
            Price.append(wines[i][1])
      for i in range(len(wines)):
            FloorArea.append(wines[i][2])
      for i in range(len(wines)):
            NoOfBedrooms.append(wines[i][3])
      for i in range(len(wines)):
            NoOfBathrooms.append(wines[i][4])

      # Finding error without normalisation.
      error, W = findW(FloorArea, NoOfBedrooms ,NoOfBathrooms, Price)
      print("W =", W)
      print("% Error in testing data set : ", error * 100)
      
      print("\n")

      # Now finding error with normalisation
      error, W = findWAfterNormalisation(FloorArea, NoOfBedrooms ,NoOfBathrooms, Price)
      print("W =", W)
      print("% Error after normalisation: ", error*100)

if __name__ == "__main__":
      main()