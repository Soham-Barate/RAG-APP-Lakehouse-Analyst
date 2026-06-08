"""
Synthetic e-commerce data generator.

Creates realistic demo data for testing the RAG pipeline:
- 1000+ products across 10 categories
- 50,000 orders
- 10,000 customer profiles
- 20,000 product reviews
- 5,000 support tickets
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data templates
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors",
    "Books", "Beauty & Personal Care", "Toys & Games",
    "Automotive", "Food & Grocery", "Health & Wellness",
]

PRODUCT_TEMPLATES = {
    "Electronics": [
        ("TechPro Wireless Earbuds", "Premium wireless earbuds with active noise cancellation, 30-hour battery life, and IPX5 water resistance. Features Bluetooth 5.3, custom EQ via app, and crystal-clear microphone for calls."),
        ("SmartFit Watch Pro", "Advanced fitness smartwatch with GPS, heart rate monitoring, blood oxygen tracking, and 7-day battery. Compatible with iOS and Android. Features 100+ workout modes and sleep tracking."),
        ("UltraView 4K Monitor 27\"", "Professional-grade 27-inch 4K UHD monitor with 100% sRGB, HDR400, USB-C connectivity, and adjustable stand. Perfect for creative professionals and gamers."),
        ("PowerMax Laptop 15", "High-performance laptop with Intel i7, 32GB RAM, 1TB SSD, and NVIDIA RTX 4060. 15.6\" FHD display with 144Hz refresh rate. Perfect for gaming and content creation."),
        ("SoundBar Elite 5.1", "Immersive 5.1 channel soundbar system with wireless subwoofer. Dolby Atmos support, HDMI eARC, and multi-room audio capability."),
    ],
    "Clothing": [
        ("Classic Fit Oxford Shirt", "Premium cotton Oxford shirt in slim fit. Wrinkle-resistant fabric, button-down collar, and chest pocket. Available in 8 colors."),
        ("UltraComfort Running Shoes", "Lightweight running shoes with responsive cushioning, breathable mesh upper, and durable rubber outsole. Great for daily runs and gym workouts."),
        ("All-Weather Jacket Pro", "Waterproof and windproof jacket with sealed seams, adjustable hood, and breathable membrane. Perfect for hiking and everyday wear."),
        ("Luxury Cashmere Sweater", "100% cashmere crew neck sweater with ribbed cuffs and hem. Soft, warm, and incredibly comfortable for fall and winter."),
        ("Performance Yoga Pants", "High-waist yoga pants with 4-way stretch, moisture-wicking fabric, and hidden pocket. Squat-proof and perfect for all activities."),
    ],
    "Home & Garden": [
        ("UltraClean Air Purifier", "HEPA H13 air purifier for rooms up to 500 sq ft. Captures 99.97% of particles, with smart auto mode, air quality sensor, and whisper-quiet operation."),
        ("SmartBrew Coffee Maker", "12-cup programmable coffee maker with built-in grinder, temperature control, and WiFi connectivity. Schedule brews from your phone."),
        ("Ergonomic Office Chair Pro", "Premium ergonomic office chair with lumbar support, adjustable armrests, headrest, and breathable mesh back. Supports up to 300 lbs."),
        ("Robot Vacuum Navigator X", "AI-powered robot vacuum with LiDAR navigation, 3000Pa suction, self-emptying base, and multi-floor mapping. Works with Alexa and Google Home."),
        ("Indoor Herb Garden Kit", "Smart indoor garden system with LED grow lights, automatic watering, and space for 6 pods. Includes basil, cilantro, and mint seed pods."),
    ],
}

ORDER_STATUSES = ["delivered", "shipped", "processing", "cancelled", "returned"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"]
SHIPPING_METHODS = ["standard", "express", "overnight", "free"]

REVIEW_SENTIMENTS = {
    "positive": [
        "Absolutely love this product! {detail}. Would definitely recommend to friends and family.",
        "Exceeded my expectations. {detail}. Great value for the price.",
        "Best purchase I've made this year. {detail}. Five stars without hesitation!",
        "Outstanding quality. {detail}. Customer service was also excellent.",
    ],
    "neutral": [
        "Decent product for the price. {detail}. Nothing special but gets the job done.",
        "Mixed feelings. {detail}. It works fine but could be better in some areas.",
        "It's okay. {detail}. Met my basic needs but won't blow you away.",
    ],
    "negative": [
        "Disappointed with this purchase. {detail}. Not worth the price in my opinion.",
        "Product arrived damaged and {detail}. The return process was also frustrating.",
        "Does not match the description. {detail}. Would not buy again.",
    ],
}

REVIEW_DETAILS = {
    "positive": [
        "The build quality is exceptional",
        "Battery life lasts way longer than advertised",
        "Setup was incredibly easy",
        "Works perfectly right out of the box",
        "The design is sleek and modern",
    ],
    "neutral": [
        "The instructions could be clearer",
        "Takes a while to get used to",
        "Size is slightly different than expected",
        "Works well but the app needs improvement",
    ],
    "negative": [
        "Stopped working after 2 weeks",
        "The quality feels cheap compared to competitors",
        "Customer support was unresponsive",
        "Multiple defects found upon delivery",
    ],
}

TICKET_TEMPLATES = [
    ("Order not received", "I placed order #{order_id} on {date} and it still hasn't arrived. The tracking shows {status}. Can you help me locate my package?"),
    ("Defective product", "The {product} I received is defective. {issue}. I'd like a replacement or refund. Order #{order_id}."),
    ("Return request", "I'd like to return the {product} from order #{order_id}. {reason}. Please provide a return shipping label."),
    ("Billing issue", "I was charged twice for order #{order_id} totaling ${amount}. Please refund the duplicate charge."),
    ("Product inquiry", "I'm interested in the {product}. Can you tell me if {question}? Also, when will it be back in stock?"),
]


class SyntheticDataGenerator:
    """Generates realistic e-commerce demo data."""

    def __init__(self, output_dir: str = "data") -> None:
        self.output_dir = Path(output_dir)
        random.seed(42)  # Reproducible data

    def generate_all(self) -> Path:
        """Generate all synthetic data and write to output directory."""
        logger.info("Generating synthetic e-commerce data...")

        # Create directory structure
        for layer in ["bronze", "silver", "gold"]:
            (self.output_dir / layer).mkdir(parents=True, exist_ok=True)

        # Generate data
        products = self._generate_products(1000)
        customers = self._generate_customers(10000)
        orders = self._generate_orders(50000, products, customers)
        reviews = self._generate_reviews(20000, products, customers)
        tickets = self._generate_tickets(5000, products, orders)

        # Write as JSON files
        self._write_json(self.output_dir / "gold" / "products.json", products)
        self._write_json(self.output_dir / "silver" / "customers.json", customers)
        self._write_json(self.output_dir / "silver" / "orders.json", orders)
        self._write_json(self.output_dir / "bronze" / "reviews.json", reviews)
        self._write_json(self.output_dir / "bronze" / "tickets.json", tickets)

        logger.info(
            "Generated: %d products, %d customers, %d orders, "
            "%d reviews, %d tickets",
            len(products), len(customers), len(orders),
            len(reviews), len(tickets),
        )

        return self.output_dir

    def _generate_products(self, count: int) -> list[dict[str, Any]]:
        """Generate product catalog."""
        products: list[dict[str, Any]] = []

        for i in range(count):
            category = random.choice(CATEGORIES)
            templates = PRODUCT_TEMPLATES.get(category, PRODUCT_TEMPLATES["Electronics"])
            name_template, desc_template = random.choice(templates)

            # Add variation to names
            variant = f" v{random.randint(1, 5)}.{random.randint(0, 9)}"
            name = name_template + (variant if i >= len(templates) * len(CATEGORIES) else "")

            products.append({
                "product_id": f"PROD-{i+1:05d}",
                "name": name,
                "category": category,
                "description": desc_template,
                "specifications": f"Weight: {random.uniform(0.1, 15):.1f}kg, Dimensions: {random.randint(5,50)}x{random.randint(5,50)}x{random.randint(2,30)}cm",
                "price": round(random.uniform(9.99, 999.99), 2),
                "rating": round(random.uniform(2.5, 5.0), 1),
                "review_count": random.randint(0, 5000),
                "sku": f"SKU-{uuid.uuid4().hex[:8].upper()}",
                "in_stock": random.random() > 0.1,
                "department": random.choice(["Retail", "Wholesale", "Online Exclusive"]),
            })

        return products

    def _generate_customers(self, count: int) -> list[dict[str, Any]]:
        """Generate customer profiles."""
        customers: list[dict[str, Any]] = []
        for i in range(count):
            customers.append({
                "customer_id": f"CUST-{i+1:06d}",
                "tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
                "total_orders": random.randint(1, 200),
                "total_spent": round(random.uniform(50, 50000), 2),
                "join_date": (
                    datetime(2020, 1, 1) + timedelta(days=random.randint(0, 2000))
                ).isoformat(),
                "preferred_category": random.choice(CATEGORIES),
            })
        return customers

    def _generate_orders(
        self, count: int, products: list, customers: list
    ) -> list[dict[str, Any]]:
        """Generate order records."""
        orders: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            customer = random.choice(customers)
            qty = random.randint(1, 5)
            total = round(product["price"] * qty, 2)
            order_date = datetime(2024, 1, 1) + timedelta(
                days=random.randint(0, 900)
            )

            orders.append({
                "order_id": f"ORD-{i+1:07d}",
                "customer_id": customer["customer_id"],
                "product_id": product["product_id"],
                "product_name": product["name"],
                "quantity": qty,
                "total_amount": total,
                "status": random.choice(ORDER_STATUSES),
                "order_date": order_date.isoformat(),
                "payment_method": random.choice(PAYMENT_METHODS),
                "shipping_method": random.choice(SHIPPING_METHODS),
                "item_count": qty,
            })
        return orders

    def _generate_reviews(
        self, count: int, products: list, customers: list
    ) -> list[dict[str, Any]]:
        """Generate product reviews with varied sentiment."""
        reviews: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            customer = random.choice(customers)

            # Weighted sentiment distribution: 60% pos, 25% neutral, 15% neg
            sentiment = random.choices(
                ["positive", "neutral", "negative"],
                weights=[60, 25, 15],
            )[0]

            template = random.choice(REVIEW_SENTIMENTS[sentiment])
            detail = random.choice(REVIEW_DETAILS[sentiment])
            text = template.format(detail=detail)

            rating_map = {"positive": (4, 5), "neutral": (3, 3), "negative": (1, 2)}
            rating = random.randint(*rating_map[sentiment])

            reviews.append({
                "review_id": f"REV-{i+1:06d}",
                "product_id": product["product_id"],
                "product_name": product["name"],
                "customer_id": customer["customer_id"],
                "rating": rating,
                "review_text": text,
                "sentiment": sentiment,
                "verified_purchase": random.random() > 0.2,
                "helpful_votes": random.randint(0, 500),
                "review_date": (
                    datetime(2024, 1, 1) + timedelta(days=random.randint(0, 900))
                ).isoformat(),
                "category": product["category"],
            })
        return reviews

    def _generate_tickets(
        self, count: int, products: list, orders: list
    ) -> list[dict[str, Any]]:
        """Generate support tickets."""
        tickets: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            order = random.choice(orders)
            template = random.choice(TICKET_TEMPLATES)

            title = template[0]
            body = template[1].format(
                order_id=order["order_id"],
                product=product["name"],
                date=order["order_date"][:10],
                status=order["status"],
                issue="the screen has dead pixels" if "defective" in title.lower() else "it doesn't match the description",
                reason="The size doesn't fit" if "return" in title.lower() else "Changed my mind",
                amount=order["total_amount"],
                question="it's compatible with USB-C",
            )

            tickets.append({
                "ticket_id": f"TKT-{i+1:06d}",
                "title": title,
                "body": body,
                "priority": random.choice(["low", "medium", "high", "urgent"]),
                "status": random.choice(["open", "in_progress", "resolved", "closed"]),
                "category": random.choice(["shipping", "product", "billing", "returns", "inquiry"]),
                "product_id": product["product_id"],
                "order_id": order["order_id"],
                "created_at": (
                    datetime(2024, 6, 1) + timedelta(days=random.randint(0, 700))
                ).isoformat(),
                "department": random.choice(["Support", "Returns", "Billing", "Shipping"]),
            })
        return tickets

    @staticmethod
    def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
        """Write data as a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Wrote %d records to %s", len(data), path)
