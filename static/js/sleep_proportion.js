
var chartDom = document.getElementById('sleep_proportion');
var sleepProportionChart = echarts.init(chartDom);
var option;

// 从服务器获取数据
fetch('/sleep-data') // 发起请求到 /sleep-data 路由
  .then(response => response.json())
  .then(data => {
    option.series[0].data = data; // 使用获取到的数据
    sleepProportionChart.setOption(option); // 更新图表
  })
  .catch(error => console.error('Error loading the sleep data:', error));


option = {
  tooltip: {
    trigger: 'item',
    formatter: '{a} <br/>{b} : {c}min ({d}%)'
  },
  series: [
    {
      name: 'sleep_proportion',
      type: 'pie',
      radius: '50%',
      data: [],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
};



// 使用刚指定的配置项和数据显示图表。
option && sleepProportionChart.setOption(option);
