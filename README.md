# IMPORTANT
As of Sept 21, 2019, Coach v2 models are no longer supported by this .NET SDK as they are built with TF 1.14. This project uses [TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp), which currently only supports up to TF 1.13. Plans to update TensorFlowSharp or potentially migrate to ML.NET are in-progress. Thank you for your patience! ✌️

# Coach .NET Core SDK

Coach is an end-to-end Image Recognition platform, we provide the tooling to do effective data collection, training, and on-device parsing of Image Recognition models.

The .NET SDK interacts with the Coach web API in order to download and parse your trained Coach models.

## Installation
Install and update from NuGet
```bash
dotnet add package Coach-ML
```

## Usage

Coach can be initialized 2 different ways. If you are only using the offline model parsing capabilities and already have a model package on disk, you can initialize like so:

```csharp
var coach = new CoachClient();

// We already had the `flowers` model on disk, no need to authenticate:
var prediction = coach.GetModel("flowers").Predict("rose.jpg").Best();
Console.WriteLine($"{prediction.Label}: {prediction.Confidence}");
```

However, in order to download your trained models, you must authenticate with your API key:
```csharp
var coach = new CoachClient().Login("myapikey");

// Now that we're authenticated, we can cache our models for future use:
await coach.CacheModel("flowers");

// Evaluate with our cached model:
var results = coach.GetModel("flowers").Predict("rose.jpg");
var bestMatch = results.Best();
```

Another, more concise example not using caching:
```csharp
var coach = new CoachClient().Login("myapikey");
var prediction = await coach.GetModelRemote("flowers").Predict("rose.jpg").Best();
```

## API Breakdown

### CoachClient
`CoachClient(bool isDebug = false)`  
Optional `isDebug`, if `true`, additional logs will be displayed.

`async Task<CoachClient> Login(string apiKey)`  
Authenticates with Coach service and allows for model caching. Accepts API Key as its only parameter. Returns its instance of `CoachClient`.

`async Task CacheModel(string name, string path=".")`  
Downloads model from Coach service to disk. Specify the name of the model, and the path to store it. This will create a new directory in the specified path and store any model related documents there. By default it will skip the download if the local version of the model matches the remote.

`CoachModel GetModel(string modelName, string path=".")`  
Loads model into memory. Specify the name and path of the model.

`async Task<CoachModel> GetModelRemote(string name, string path=".")`  
Downloads model from Coach service to disk, and loads it into memory.

### CoachModel
`CoachModel(TFGraph graph, string[] labels, string module, float coachVersion)`  
Initializes a new instance of `CoachModel`.

`CoachResult Predict(string image, string inputName = "input", string outputName = "output")`  
Specify the directory of an image file. Parses the specified image as a Tensor and runs it through the loaded model. Optionally accepts `input` and `output` tensor names.

`CoachResult Predict(byte[] image, string inputName = "input", string outputName = "output")`  
Specify the image as a byte array. Parses the specified image as a Tensor and runs it through the loaded model. Optionally accepts `input` and `output` tensor names.

### CoachResult
`List<LabelProbability> Results`  
Unsorted prediction results.

`List<LabelProbability> SortedResults`  
Sorted prediction results, descending in Confidence.

`LabelProbability Best()`  
Most Confident result.

`LabelProbability Worst()`  
Least Confident result.

### LabelProbability
`string Label` -> Label of result

`float Confidence` -> Confidence of result
