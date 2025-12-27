
// 1. Hàm load dữ liệu Trending
async function loadTrendingItems() {
    try {
        const res = await fetch('/api/popular-items');
        const items = await res.json();
        
        const container = document.getElementById('trending-container');
        container.innerHTML = ''; // Xóa loading

        items.forEach((item, index) => {
            // Tạo HTML cho từng card
            const cardHtml = `
                <div class="card trending-card shadow-sm border-0">
                    <div class="position-relative">
                        <img src="${item.image_url}" class="card-img-top" style="height: 140px; object-fit: cover;">
                        <div class="position-absolute top-0 start-0 bg-danger text-white px-2 py-1 small fw-bold" 
                                style="border-bottom-right-radius: 10px;">
                            #${index + 1}
                        </div>
                    </div>
                    <div class="card-body p-2">
                        <h6 class="card-title text-truncate" title="${item.title}">${item.title}</h6>
                        <p class="card-text small text-muted mb-1">
                            <i class="bi bi-eye-fill"></i> ${item.metrics?.click_count || 0} views
                        </p>
                            <button class="btn btn-sm btn-outline-primary w-100" onclick="window.location.href='/item/${item.item_id}'">
                            Xem ngay
                        </button>
                    </div>
                </div>
            `;
            container.innerHTML += cardHtml;
        });

    } catch (error) {
        console.error("Lỗi load trending:", error);
    }
}

// 2. Hàm xử lý nút bấm qua lại
function scrollTrending(direction) {
    const container = document.getElementById('trending-container');
    const scrollAmount = 300; // Mỗi lần bấm trượt 300px
    
    if (direction === 1) {
        container.scrollLeft += scrollAmount;
    } else {
        container.scrollLeft -= scrollAmount;
    }
}

// Gọi hàm khi trang web tải xong
document.addEventListener('DOMContentLoaded', () => {
    loadTrendingItems();
    // renderItems(); // Hàm render chính cũ của bạn
});
