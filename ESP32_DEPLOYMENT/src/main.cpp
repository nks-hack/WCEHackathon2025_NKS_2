
#include "model_data.h" // The header file containing your model data
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <EEPROM.h>

// Tensor arena size (adjust based on your model's RAM requirement)
constexpr int tensor_arena_size = 10 * 1024; // 16 KB
uint8_t tensor_arena[tensor_arena_size];

// Error reporter
tflite::MicroErrorReporter error_reporter;

// Load the model from the C++ array
const tflite::Model* model = tflite::GetModel(quantized_model_int8_tflite);

// Use all ops resolver
tflite::AllOpsResolver resolver;

// Create the interpreter
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, &error_reporter);

// Standard Scaler Parameters
const float mean[5] = {252.25, 28.22, 72.85, 32.02, 227.78};
const float scale[5] = {6.39, 7.23, 5.37, 10.32, 4.05};

void setup() {
  Serial.begin(9600);
  // Allocate tensors
  interpreter.AllocateTensors();
}

void run_inference() {
  // Get input and output tensors
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // Raw input features: op1, temp, thi, ozone_lag1, op2
  float input_features[5] = {256, 28.2, 72.41, 33.17, 228};

  // Standardize the input
  for (int i = 0; i < 5; i++) {
    input_features[i] = (input_features[i] - mean[i]) / scale[i];
  }

  // Assign standardized input to the model
  for (int i = 0; i < 5; i++) {
    input->data.f[i] = input_features[i];
  }

  // Run inference
  interpreter.Invoke();

  // Retrieve the output (dequantize to float if needed)
  int8_t raw_output = output->data.int8[0];
  float scale = output->params.scale;  // From quantization params
  int32_t zero_point = output->params.zero_point;

  float final_output = (raw_output - zero_point) * scale;

  // Print the output
  Serial.print("Prediction: ");
  Serial.println(final_output);
}

void loop() {
  run_inference();
  delay(1000); // Run inference every second
}