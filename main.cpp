#include <Arduino.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h"

#define TENSOR_ARENA_SIZE 32 * 1024 
uint8_t tensor_arena[TENSOR_ARENA_SIZE];


tflite::MicroMutableOpResolver<10> resolver;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    Serial.println("Initializing TensorFlow Lite Model...");

    Serial.printf("Free Heap: %d\n", ESP.getFreeHeap());


    Serial.println("Loading model...");
    model = tflite::GetModel(quantized_model_tflite);
    if (model == nullptr) {
        Serial.println("Error: Model is NULL! Check model.h file.");
        return;
    }


    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model schema version mismatch! Expected: ");
        Serial.print(TFLITE_SCHEMA_VERSION);
        Serial.print(", but got: ");
        Serial.println(model->version());
        return;
    }


    Serial.println("Adding operators...");
    resolver.AddFullyConnected(); 
    resolver.AddRelu();         
    resolver.AddSoftmax();         



    Serial.println("Creating interpreter...");
    interpreter = new tflite::MicroInterpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE,
        nullptr  
    );
    

    Serial.println("Allocating tensors...");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed! Check memory constraints.");
        return;
    }

    Serial.println("Model initialized successfully.");
}

void loop() {
    Serial.println("Running inference...");


    float input_data[5] = {256.0, 28.2, 0.8 * 28.2 + (25.0 * (28.2 - 14.4)) / 100 + 46.4, 33.17, 228.0};

    TfLiteTensor *input = interpreter->input(0);
    if (!input) {
        Serial.println("Error: Input tensor is NULL!");
        return;
    }

    for (int i = 0; i < 5; i++) {
        input->data.f[i] = input_data[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed! Check tensor allocation.");
        return;
    }


    TfLiteTensor *output = interpreter->output(0);
    if (!output) {
        Serial.println("Error: Output tensor is NULL!");
        return;
    }

    float predicted_value = output->data.f[0];
    Serial.print("Predicted Value: ");
    Serial.println(predicted_value);

    delay(5000);
}