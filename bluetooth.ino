#include <SoftwareSerial.h>                  // Serial 통신을 하기 위해 선언

SoftwareSerial HM10 (10, 9); // HM-10모듈 10=RXD , 9=TXD 핀 선언
//SoftwareSerial STM (2, 3); //2=RX, 3=TX
int a;
int xPos;
int yPos;

void setup() 
{

    Serial.begin(9600);                            // 시리얼 통신 선언 (보드레이트 9600)

    HM10.begin(9600);                             // HM-10 모듈 통신 선언 (보드레이트 9600)
    
    //STM.begin(115200);                           // STM 모듈 통신 선언 (보드레이트 115200)
}
void loop() 
{
    //if(STM.available())
      //Serial.write(STM.read());
      //Serial.print("\n");

    if(HM10.available())                           // HM-10에 입력이 되면
  
      Serial.write(HM10.read());               // HM-10에 입력된 값을 시리얼 모니터에 출력

    if(Serial.available())                           // 시리얼 모니터에 입력이 되면

      HM10.write(Serial.read());              // 그 값을 HM-10에 출력
    a = analogRead(A0);
    xPos = analogRead(A1);
    yPos = analogRead(A2);
    /*Serial.print(a);
    Serial.print("  ");
    Serial.print(xPos);
    Serial.print("  ");
    Serial.print(yPos);
    Serial.print("\n");*/
    if(yPos == 0) {
      if(a < 200) {
        HM10.write("0");
      } else if (a < 600) {
        HM10.write("3");
      } else {
        HM10.write("4");
      }
    } else if (yPos == 1023) {
      if(a < 200) {
      HM10.write("0");
      } else if (a < 600) {
        HM10.write("5");
      } else {
        HM10.write("6");
      }
    } else {
      if(a < 200) {
        HM10.write("0");
      } else if (a < 600) {
        HM10.write("1");
      } else {
        HM10.write("2");
      } 
    }
    //HM10.write(a);
    delay(10);
}