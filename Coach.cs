using System;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Net.Http;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using TensorFlow;

namespace Coach {
    public class CoachModel {
        public CoachModel(TFGraph graph, string[] labels, string module) {

        }
        
    }

    public struct Model {
        public string name { get; set; }
        public string status { get; set; }
        public int version { get; set; }
        public string module { get; set; }
        public string[] labels { get; set; }
    }

    public class Profile {
        [JsonProperty("id")]
        public string Id { get; set; }
        [JsonProperty("bucket")]
        public string Bucket { get; set; }

        [JsonProperty("models")]
        public Model[] Models { get; set; }
    }

    public class CoachClient {
        
        public bool IsDebug { get; private set; }
        private Profile Profile { get; set; }
        private string ApiKey { get; set; }

        public CoachClient(bool isDebug = false) {
            this.IsDebug = isDebug;
        }

        public async Task Login(string apiKey) {
            this.ApiKey = apiKey;
            this.Profile = await GetProfile();
        }

        private bool IsAuthenticated() {
            return this.Profile != null;
        }

        private Model ReadManifest(string path) {
            return JsonConvert.DeserializeObject<Model>(File.ReadAllText(path));
        }

        public async Task CacheModel(string name, string path=".") {
            if (!IsAuthenticated())
                throw new Exception("User is not authenticated");

            // Create the target dir
            if (!Directory.Exists(name))
                Directory.CreateDirectory(name);

            Model model = this.Profile.Models.Single(m => m.name == name);

            int profileVersion = model.version;
            string profileManifest = $"{path}/{name}/manifest.json";

            if (File.Exists(profileManifest)) {
                // Load existing model manifest
                Model manifest = ReadManifest(profileManifest);
                int manifestVersion = manifest.version;

                if (profileVersion == manifestVersion) {
                    if (this.IsDebug) {
                        Console.WriteLine("Version match, skipping model download");
                    }

                    return;
                }
            } else {
                var json = JsonConvert.SerializeObject(model);
                File.WriteAllText(profileManifest, json);
            }

            var baseUrl = $"https://la41byvnkj.execute-api.us-east-1.amazonaws.com/prod/{this.Profile.Bucket}/model-bin?object=trained/{name}/{profileVersion}/model";
                    
            var modelFile = "frozen.pb";
            var modelUrl = $"{baseUrl}/{modelFile}";


            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);
            request.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/octet-stream"));
            
            var requestMessage = new HttpRequestMessage(HttpMethod.Get, modelUrl);
            requestMessage.Content = new StringContent(String.Empty, Encoding.UTF8, "application/octet-stream");
            
            var response = await request.SendAsync(requestMessage);
            response.EnsureSuccessStatusCode();

            byte[] modelBytes = await response.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes($"{path}/{name}/{modelFile}", modelBytes);
        }

        private async Task<Profile> GetProfile() {
            var id = this.ApiKey.Substring(0, 5);
            var url = $"https://2hhn1oxz51.execute-api.us-east-1.amazonaws.com/prod/{id}";

            var request = new HttpClient();
            request.DefaultRequestHeaders.Add("X-Api-Key", this.ApiKey);

            var response = await request.GetAsync(url);
            response.EnsureSuccessStatusCode();

            string responseBody = await response.Content.ReadAsStringAsync();
            var profile = JsonConvert.DeserializeObject<Profile>(responseBody);
            return profile;
        }

        public CoachModel GetModel(string path) {
            var graphPath = $"{path}/frozen.pb";
            var labelPath = $"{path}/manifest.json";
            
            // Load the graphdef
            var graph = new TFGraph();
            graph.Import(File.ReadAllBytes(graphPath));

            var manifest = ReadManifest(labelPath);
            
            string[] labels = manifest.labels;
            string baseModule = manifest.module;

            return new CoachModel(graph, labels, baseModule);
        }
    }
}