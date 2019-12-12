using Coach;
using System;
using Xunit;

namespace Coach.Tests
{
    public class ModelTests
    {
        private readonly CoachModel _model;

        public ModelTests()
        {
            var c = new CoachClient(isDebug: true).Login("").Result;
            _model = c.GetModelRemote("small-flowers").Result;
        }

        [Fact]
        public void Predict()
        {
            var preciction = _model.Predict("rose.jpg");
            Assert.Equal("rose", preciction.Best().Label);
        }
    }
}
