document.addEventListener('DOMContentLoaded', function() {
    const attendBtn = document.querySelector('.btn:nth-child(1)');
    const addUserBtn = document.querySelector('.btn:nth-child(2)');
    const showAttendanceBtn = document.querySelector('.btn:nth-child(3)');

    attendBtn.addEventListener('click', function() {
        window.location.href = '/attend.html';
    });

    addUserBtn.addEventListener('click', function() {
        window.location.href = '/adduser.html';
    });

    showAttendanceBtn.addEventListener('click', function() {
        window.location.href = '/showattendance.html';
    });
});