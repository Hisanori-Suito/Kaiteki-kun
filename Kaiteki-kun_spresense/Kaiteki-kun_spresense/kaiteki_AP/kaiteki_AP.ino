#include <Audio.h>
#include <FFT.h>
#include <SDHCI.h>
SDClass SD;
#define FFT_LEN 1024
#define AVG_FILTER (8)
FFTClass<AS_CHANNEL_MONO, FFT_LEN> FFT; 
AudioClass *theAudio = AudioClass::getInstance();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define  AP_SSID        "GS2200_LIMITED_AP"
#define  PASSPHRASE     "123456789"
#define  AP_CHANNEL     6
#define  TCPSRVR_PORT   "80"

#include <GS2200Hal.h>
#include <GS2200AtCmd.h>
#include <TelitWiFi.h>

#include <Camera.h>
#include <time.h>

#define  CONSOLE_BAUDRATE  115200

/*タイマー関連*/
#define INTERVAL 100
static unsigned long counter = 0;
static unsigned long counter_ms = 0;
static unsigned char audioflag = 0;
static unsigned char beepstate = 0;
static long beepcnt = 0;
static long miccnt = 0;
unsigned int callback_func() {
  if (counter>100 && audioflag==0 && beepstate<=1){
    static const uint32_t buffer_size2 = FFT_LEN*sizeof(int16_t);
    static char buff2[buffer_size2];
    uint32_t read_size2;
    int ret2 = theAudio->readFrames(buff2, buffer_size2, &read_size2);
    counter=0;
  }
  ++counter;
  ++counter_ms;
  if(beepcnt>=20){
    ++beepcnt;
  }
  if(miccnt>=20){
    ++miccnt;
  }
  return INTERVAL;
}
/*タイマー関連*/
extern uint8_t ESCBuffer[];
extern uint32_t ESCBufferCnt;

TelitWiFi gs2200;
TWIFI_Params gsparams;

int               g_width   = CAM_IMGSIZE_VGA_H;
int               g_height  = CAM_IMGSIZE_VGA_V;
CAM_IMAGE_PIX_FMT g_img_fmt = CAM_IMAGE_PIX_FMT_JPG;
int               g_divisor = 7;

// Contents data creation
String header         = " HTTP/1.1 200 OK \r\n";
String content        = "<!DOCTYPE html><html> Test!</html>\r\n";
String content_type   = "Content-type: text/html \r\n";
String content_length = "Content-Length: " + String(content.length()) + "\r\n\r\n";

CAM_WHITE_BALANCE g_wb      = CAM_WHITE_BALANCE_AUTO;
int send_contents(char* ptr, int size)
{  
  String str = header + content_type + content_length + content;
  size = (str.length() > size)? size : str.length();
  str.getBytes(ptr, size);
  return str.length();
}

void printError(enum CamErr err)
{
  Serial.print("Error: ");
  switch (err) {
  case CAM_ERR_NO_DEVICE:             Serial.println("No Device");                     break;
  case CAM_ERR_ILLEGAL_DEVERR:        Serial.println("Illegal device error");          break;
  case CAM_ERR_ALREADY_INITIALIZED:   Serial.println("Already initialized");           break;
  case CAM_ERR_NOT_INITIALIZED:       Serial.println("Not initialized");               break;
  case CAM_ERR_NOT_STILL_INITIALIZED: Serial.println("Still picture not initialized"); break;
  case CAM_ERR_CANT_CREATE_THREAD:    Serial.println("Failed to create thread");       break;
  case CAM_ERR_INVALID_PARAM:         Serial.println("Invalid parameter");             break;
  case CAM_ERR_NO_MEMORY:             Serial.println("No memory");                     break;
  case CAM_ERR_USR_INUSED:            Serial.println("Buffer already in use");         break;
  case CAM_ERR_NOT_PERMITTED:         Serial.println("Operation not permitted");       break;
  default:
    break;
  }
  exit(1);
}

// the setup function runs once when you press reset or power the board
void setup() {

  /* initialize digital pin of LEDs as an output. */
  pinMode(LED0, OUTPUT);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);

  digitalWrite( LED0, HIGH );   // turn the LED off (LOW is the voltage level)
  Serial.begin( CONSOLE_BAUDRATE ); // talk to PC

  /* Initialize SPI access of GS2200 */
  //Init_GS2200_SPI();
  Init_GS2200_SPI_type(iS110B_TypeC);

  /* Initialize AT Command Library Buffer */
  gsparams.mode = ATCMD_MODE_LIMITED_AP;
  //gsparams.mode = ATCMD_MODE_STATION;
  gsparams.psave = ATCMD_PSAVE_DEFAULT;
  if( gs2200.begin( gsparams ) ){
    ConsoleLog( "GS2200 Initilization Fails" );
    while(1);
  }
///////////////////////////////////////////////////////////////////
  /* GS2200 runs as AP */
  if( gs2200.activate_ap( AP_SSID, PASSPHRASE, AP_CHANNEL ) ){
    ConsoleLog( "WiFi Network Fails" );
    while(1);
  }
    /* GS2200 Association to AP */
  /*if( gs2200.activate_station( AP_SSID, PASSPHRASE ) ){
    ConsoleLog( "Association Fails" );
    while(1);
  }*/
  /////////////////////////////////////////////////////////////////////////
  digitalWrite( LED1, HIGH );   // turn the LED off (LOW is the voltage level)
  Serial.println("Setup Camera...");

  CamErr err = theCamera.begin();
  if (err != CAM_ERR_SUCCESS) { printError(err); }
  err = theCamera.setAutoWhiteBalanceMode(g_wb);
  if (err != CAM_ERR_SUCCESS) { printError(err); }
  err = theCamera.setStillPictureImageFormat(g_width, g_height, g_img_fmt, g_divisor);
  if (err != CAM_ERR_SUCCESS) { printError(err); }

  Serial.println("Setup Camera done.");
  digitalWrite( LED2, HIGH );   // turn the LED off (LOW is the voltage level)
  while (!SD.begin()) { Serial.println("Insert SD card"); }
  FFT.begin(WindowHamming, AS_CHANNEL_MONO, (FFT_LEN/2));
  Serial.println("Init Audio Recorder");
  theAudio->begin();
  theAudio->setReadyMode();
  theAudio->setPlayerMode(AS_SETPLAYER_OUTPUTDEVICE_SPHP, 0, 0);
  theAudio->setBeep(1,-40,440);
  usleep(1000 * 1000);

  // 入力をマイクに設定
  digitalWrite( LED3, HIGH );   // turn the LED off (LOW is the voltage level)
  
  theAudio->setReadyMode();
  theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);
  int erraud = theAudio->initRecorder(AS_CODECTYPE_PCM 
    ,"/mnt/sd0/BIN", AS_SAMPLINGRATE_48000, AS_CHANNEL_MONO);                             
  if (erraud != AUDIOLIB_ECODE_OK) {
    Serial.println("Recorder initialize error");
    while(1);
  }
  Serial.println("Start Recorder");
  theAudio->startRecorder(); // 録音開始
  attachTimerInterrupt(callback_func, INTERVAL);
  digitalWrite( LED0, LOW ); // turn on LED
  digitalWrite( LED1, LOW ); // turn on LED
  digitalWrite( LED2, LOW ); // turn on LED
  digitalWrite( LED3, LOW ); // turn on LED
}

// the loop function runs over and over again forever
void loop() {
  ATCMD_RESP_E resp;
  char server_cid = 0, remote_cid=0;
  ATCMD_NetworkStatus networkStatus;
  uint32_t timer=0;

  resp = ATCMD_RESP_UNMATCH;
  ConsoleLog( "Start TCP Server");
  
  resp = AtCmd_NSTCP( TCPSRVR_PORT, &server_cid);
  
  if (resp != ATCMD_RESP_OK) {
    ConsoleLog( "No Connect!" );
    delay(2000);
    return;
  }

  if (server_cid == ATCMD_INVALID_CID) {
    ConsoleLog( "No CID!" );
    delay(2000);
    return;
  }

  float Value=0.0;
  while( 1 ){
    digitalWrite( LED0, !digitalRead(LED0) );   // turn the LED off (LOW is the voltage level)
    ConsolePrintf( ":%d:", beepcnt);
    if(Value>0.07 && miccnt<20 && beepcnt==0){
      ConsolePrintf("うるさい\r\n");
      ConsolePrintf( ":%lf:", Value);
      digitalWrite( LED3, HIGH );   // turn the LED off (LOW is the voltage level)
      if(beepstate==0){
        beepstate=4;
      }
    }else{
      ConsolePrintf("静か\r\n");
      digitalWrite( LED3, LOW );   // turn the LED off (LOW is the voltage level)
    }
    if(beepcnt==0 && beepstate==2){
      theAudio->stopRecorder();
      theAudio->setReadyMode();
      theAudio->setPlayerMode(AS_SETPLAYER_OUTPUTDEVICE_SPHP, 0, 0);
      theAudio->setBeep(1,-30,int(2349/2));
      beepcnt=20;
    }else if(beepcnt==0 && beepstate==3){
      theAudio->stopRecorder();
      theAudio->setReadyMode();
      theAudio->setPlayerMode(AS_SETPLAYER_OUTPUTDEVICE_SPHP, 0, 0);
      theAudio->setBeep(1,-30,int(2093/2));
      beepcnt=20;
    }else if(beepcnt==0 && beepstate==4){
      theAudio->stopRecorder();
      theAudio->setReadyMode();
      theAudio->setPlayerMode(AS_SETPLAYER_OUTPUTDEVICE_SPHP, 0, 0);
      theAudio->setBeep(1,-30,int(1759/2));
      beepcnt=20;
    }
    if(miccnt>60000){
      miccnt=0;
    }
    if(beepcnt>30000 && beepstate>=2){
      beepcnt=0;
      beepstate=0;
      miccnt=20;
      Value=0.0;
      theAudio->setReadyMode();
      theAudio->setRecorderMode(AS_SETRECDR_STS_INPUTDEVICE_MIC);
      int erraud = theAudio->initRecorder(AS_CODECTYPE_PCM 
        ,"/mnt/sd0/BIN", AS_SAMPLINGRATE_48000, AS_CHANNEL_MONO);                             
      if (erraud != AUDIOLIB_ECODE_OK) {
        Serial.println("Recorder initialize error");
        while(1);
      }
      Serial.println("Start Recorder");
      theAudio->startRecorder(); // 録音開始
    }else if (beepstate==0){
      Value=voicecatch();
    }
    ConsoleLog( "Waiting for TCP Client");
    if( ATCMD_RESP_TCP_SERVER_CONNECT != WaitForTCPConnection( &remote_cid, 1000 ) ){
      continue;
    }

    ConsoleLog( "TCP Client Connected");
    // Prepare for the next chunck of incoming data
    WiFi_InitESCBuffer();
    delay(50);

    unsigned long cam_before, cam_after, one_before, one_after;
    // Start the echo server
    while(Get_GPIO37Status() ){
      resp = AtCmd_RecvResponse();
      ConsoleLog( "Get_GPIO37Status");
      if( ATCMD_RESP_BULK_DATA_RX == resp){
        if( Check_CID( remote_cid ) ){
          ConsolePrintf( "Received : %s\r\n", ESCBuffer+1 );
          String message = ESCBuffer+1;
          int space1_pos = message.indexOf(' ');
          int space2_pos = message.indexOf(' ', space1_pos + 1);
          String method  = message.substring(0, space1_pos);
          String path    = message.substring(space1_pos + 1, space2_pos);
          if (method == "GET"){
            one_before = millis();
            cam_before = millis();
            CamImage img = theCamera.takePicture();
            cam_after = millis();
            ConsolePrintf( "Take Cam:%dms\n", cam_after - cam_before );
            if(img.getImgSize() != 0) {
              String response = "HTTP/1.1 200 OK\r\n"
                                "Content-Type: image/jpeg\r\n"
                                "Content-Length: " + String(img.getImgSize()) + "\r\n"
                                "\r\n";
              ATCMD_RESP_E err = AtCmd_SendBulkData( remote_cid, response.c_str(), response.length() );
              size_t sent_size = 0;
              for (size_t sent_size = 0; sent_size < img.getImgSize();) {
                size_t send_size = min(img.getImgSize() - sent_size, MAX_RECEIVED_DATA - 100);
                ATCMD_RESP_E err = AtCmd_SendBulkData( remote_cid, (uint8_t *)(img.getImgBuff() + sent_size), send_size );
                if (ATCMD_RESP_OK == err) {
                  sent_size += send_size;
                } else {
                  ConsolePrintf( "Send Bulk Error, %d\n", err );
                  delay(1000);
                  err = AtCmd_SendBulkData( remote_cid, (uint8_t *)(img.getImgBuff() + sent_size), send_size );
                  ConsolePrintf( "Send Bulk Error, %d\n", err );
                }
                delay(5);
              }
              one_after = millis();
              ConsolePrintf( "Send:%dms\n", one_after - one_before );
            } else {
              String response = "HTTP/1.1 404 Not Found";
              ATCMD_RESP_E err = AtCmd_SendBulkData( remote_cid, response.c_str(), response.length() );
              if (ATCMD_RESP_OK != err) {
                ConsolePrintf( "Send Bulk Error, %d\n", err );
                delay(2000);
                return;
              }
            }
            char test13=*(ESCBuffer+13);
            if(test13==49){
              ConsolePrintf("忘れ物あり\r\n");
              digitalWrite( LED1, HIGH );   // turn the LED off (LOW is the voltage level)
              if(beepstate==0){
                beepstate=2;
              }
            }else{
              ConsolePrintf("忘れ物なし\r\n");
              digitalWrite( LED1, LOW );   // turn the LED off (LOW is the voltage level)
            }
            char test14=*(ESCBuffer+14);
            if(test14==49){
              ConsolePrintf("混雑\r\n");
              digitalWrite( LED2, HIGH );   // turn the LED off (LOW is the voltage level)
              if(beepstate==0){
                beepstate=3;
              }
            }else{
              ConsolePrintf("非混雑\r\n");
              digitalWrite( LED2, LOW );   // turn the LED off (LOW is the voltage level)
            }
          }else if (message.substring(0, message.indexOf(' ')) == "GET") {
            int length = send_contents(ESCBuffer+1,MAX_RECEIVED_DATA);        
            ConsolePrintf( "Will send : %s\r\n", ESCBuffer+1 );
            if( ATCMD_RESP_OK != AtCmd_SendBulkData( remote_cid, ESCBuffer+1, length ) ){
              // Data is not sent, we need to re-send the data
              delay(10);
              ConsolePrintf( "Sent Error : %s\r\n", ESCBuffer+1 );
            }
          }
        }          
        WiFi_InitESCBuffer();
      }
    }
  }
}
float voicecatch(){
    //FFT>
    static const uint32_t buffering_time = 
        FFT_LEN*1000/AS_SAMPLINGRATE_48000;
    static const uint32_t buffer_size = FFT_LEN*sizeof(int16_t);
    static const int ch_index = AS_CHANNEL_MONO-1;
    static char buff[buffer_size];
    static float pDst[FFT_LEN];
    uint32_t read_size;
    // buffer_sizeで要求されたデータをbuffに格納する
    // 読み込みできたデータ量は read_size に設定される
    audioflag=1;
    delay(buffering_time); // データが蓄積されるまで待つ
    delay(buffering_time); // データが蓄積されるまで待つ
    int ret = theAudio->readFrames(buff, buffer_size, 
      &read_size);
    audioflag=0;
    if (ret != AUDIOLIB_ECODE_OK && 
        ret != AUDIOLIB_ECODE_INSUFFICIENT_BUFFER_AREA) {
      Serial.println("Error err = " + String(ret));
      theAudio->stopRecorder();
      while(1);
    }
    // 読み込みサイズがbuffer_sizeに満たない場合
    if (read_size < buffer_size) {
      delay(buffering_time); // データが蓄積されるまで待つ
      return(0.0);
    }
    FFT.put((q15_t*)buff, FFT_LEN);  // FFTを実行
    FFT.get(pDst, ch_index);  // チャンネル0番の演算結果を取得
      // 周波数と最大値の近似値を算出
    float maxValue;
    float peakFs = get_peak_frequency(pDst, &maxValue);
    Serial.print("peak freq: " + String(peakFs) + " Hz");
    Serial.print("Spectrum: " + String(maxValue));
    //<FFT
    return(maxValue);
}
float get_peak_frequency(float *pData, float* maxValue) {
  uint32_t idx;
  float delta, delta_spr;
  float peakFs;
  // 周波数分解能(delta)を算出
  const float delta_f = AS_SAMPLINGRATE_48000/FFT_LEN;
  // 最大値と最大値のインデックスを取得
  arm_max_f32(pData, FFT_LEN/2, maxValue, &idx);
  if (idx < 1) return 0.0;
  // 周波数のピークの近似値を算出
  delta = 0.5*(pData[idx-1]-pData[idx+1])
    /(pData[idx-1]+pData[idx+1]-(2.*pData[idx]));
  peakFs = (idx + delta) * delta_f;
  // スペクトルの最大値の近似値を算出
  delta_spr = 0.125*(pData[idx-1]-pData[idx+1])
    *(pData[idx-1]-pData[idx+1])
    /(2.*pData[idx]-(pData[idx-1]+pData[idx+1]));
  *maxValue += delta_spr;
  return peakFs;
}
