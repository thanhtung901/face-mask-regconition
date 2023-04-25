$(document).ready(function(){
    // Tạo 1 hàm gọi, sau 100 milisecond thì mới hiện các nội dung bên trong
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
        }, 100);
    };
    spinner();

    // Tạo thanh thay đổi điều hướng tác vụ
    $('.sidebar-toggler').click(function () {
        $('.sidebar, .content').toggleClass("open");
        return false;
    });
});

$(document).ready(function(){
    // Lấy dữ liệu thời tiết
    const getLink = "https://api.open-meteo.com/v1/forecast?latitude=21.59&longitude=105.85&hourly=temperature_2m,relativehumidity_2m,weathercode&current_weather=true&timezone=Asia%2FBangkok"
    function GetJSON(){
        URL_JSON = getLink;
        fetch(URL_JSON)
          .then(response => response.json())
          .then(Json =>{
            time = Json["current_weather"]["time"]; // Thời gian
            // console.log(time);
            var Split_Time = time.split("T")[0]; // Tách được thời gian với ngày tháng ==> Lấy được ngày tháng
            var Take_The_Time = time.split("T")[1]; // Lấy được thời gian
            var Get_Year = Split_Time.split("-")[0]; // Lấy được năm
            var Get_Month = Split_Time.split("-")[1]; // Còn lại tháng
            var Get_Day = Split_Time.slice(8); // Lấy được ngày
            document.getElementById('Time').innerHTML = Take_The_Time;
            document.getElementById('Day').innerHTML = Get_Day;
            document.getElementById('Month').innerHTML = Get_Month;
            document.getElementById('Year').innerHTML = Get_Year;
        })
    }
    setInterval(GetJSON, 300);
});