// 当文档加载完毕时执行函数
document.addEventListener('DOMContentLoaded', function() {
    // 获取当前时间并格式化
    var now = new Date();
    var formattedDate = now.getFullYear() + '-' +
                        ('0' + (now.getMonth() + 1)).slice(-2) + '-' +
                        ('0' + now.getDate()).slice(-2) + ' ' +
                        ('0' + now.getHours()).slice(-2) + ':' +
                        ('0' + now.getMinutes()).slice(-2) + ':' +
                        ('0' + now.getSeconds()).slice(-2);

    // 设置span元素的文本为当前时间
    document.querySelector('.span-dGs-1').textContent = '报告时间 : ' + formattedDate;
});