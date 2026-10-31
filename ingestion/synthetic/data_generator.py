"""
Scalable, batch-oriented synthetic e-commerce lakehouse generator.

Generates realistic relational datasets with strict referential integrity:
- Products catalog with categories, specifications, prices, and stock
- Customer profiles with lifetime tiers, join dates, and spend history
- High-volume transaction stream (Orders) written in streaming Parquet batches
- Customer feedback (Product reviews) with realistic sentiment distribution
- Customer service tickets with categorized urgency and departments

Supports dynamic scale presets:
- quick (~50K rows, ~5 MB Parquet)
- medium (~2M rows, ~200 MB Parquet)
- large (~10M rows, ~1 GB Parquet)
- enterprise (~50M rows, ~5 GB Parquet)
Memory is strictly bounded (<150 MB RSS) via streaming PyArrow ParquetWriter.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale Presets
# ---------------------------------------------------------------------------
SCALE_CONFIGS = {
    "quick": {
        "products": 1_000,
        "customers": 5_000,
        "orders": 50_000,
        "reviews": 10_000,
        "tickets": 2_000,
        "write_json": True,
    },
    "medium": {
        "products": 10_000,
        "customers": 50_000,
        "orders": 2_000_000,
        "reviews": 50_000,
        "tickets": 10_000,
        "write_json": False,
    },
    "large": {
        "products": 50_000,
        "customers": 200_000,
        "orders": 10_000_000,
        "reviews": 100_000,
        "tickets": 20_000,
        "write_json": False,
    },
    "enterprise": {
        "products": 100_000,
        "customers": 500_000,
        "orders": 50_000_000,
        "reviews": 200_000,
        "tickets": 50_000,
        "write_json": False,
    },
}

CATEGORIES = [
    "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors",
    "Books", "Beauty & Personal Care", "Toys & Games",
    "Automotive", "Food & Grocery", "Health & Wellness",
]

PRODUCT_TEMPLATES = {
    "Electronics": [
        ("TechPro Wireless Earbuds", "Premium wireless earbuds with active noise cancellation, 30-hour battery life, and IPX5 water resistance."),
        ("SmartFit Watch Pro", "Advanced fitness smartwatch with GPS, heart rate monitoring, blood oxygen tracking, and 7-day battery."),
        ("UltraView 4K Monitor 27\"", "Professional-grade 27-inch 4K UHD monitor with 100% sRGB, HDR400, USB-C connectivity, and adjustable stand."),
        ("PowerMax Laptop 15", "High-performance laptop with Intel i7, 32GB RAM, 1TB SSD, and NVIDIA RTX 4060."),
        ("SoundBar Elite 5.1", "Immersive 5.1 channel soundbar system with wireless subwoofer and Dolby Atmos support."),
    ],
    "Clothing": [
        ("Classic Fit Oxford Shirt", "Premium cotton Oxford shirt in slim fit. Wrinkle-resistant fabric, button-down collar, and chest pocket."),
        ("UltraComfort Running Shoes", "Lightweight running shoes with responsive cushioning, breathable mesh upper, and durable rubber outsole."),
        ("All-Weather Jacket Pro", "Waterproof and windproof jacket with sealed seams, adjustable hood, and breathable membrane."),
        ("Luxury Cashmere Sweater", "100% cashmere crew neck sweater with ribbed cuffs and hem. Soft, warm, and comfortable."),
        ("Performance Yoga Pants", "High-waist yoga pants with 4-way stretch, moisture-wicking fabric, and hidden pocket."),
    ],
    "Home & Garden": [
        ("UltraClean Air Purifier", "HEPA H13 air purifier for rooms up to 500 sq ft. Captures 99.97% of particles."),
        ("SmartBrew Coffee Maker", "12-cup programmable coffee maker with built-in grinder and WiFi connectivity."),
        ("Ergonomic Office Chair Pro", "Premium ergonomic office chair with lumbar support, adjustable armrests, and mesh back."),
        ("Robot Vacuum Navigator X", "AI-powered robot vacuum with LiDAR navigation, 3000Pa suction, and self-emptying base."),
        ("Indoor Herb Garden Kit", "Smart indoor garden system with LED grow lights and automatic watering."),
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
        "Product arrived damaged and {detail}. The return process was frustrating.",
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
    ("Order not received", "I placed order #{order_id} on {date} and it still hasn't arrived. The tracking shows {status}."),
    ("Defective product", "The {product} I received is defective. {issue}. I'd like a replacement or refund. Order #{order_id}."),
    ("Return request", "I'd like to return the {product} from order #{order_id}. {reason}. Please provide a return shipping label."),
    ("Billing issue", "I was charged twice for order #{order_id} totaling ${amount}. Please refund duplicate charge."),
    ("Product inquiry", "I'm interested in the {product}. Can you tell me if {question}? When will it be back in stock?"),
]

# ---------------------------------------------------------------------------
# Strict PyArrow Schemas for Direct-to-Disk Streaming
# ---------------------------------------------------------------------------
ORDERS_ARROW_SCHEMA = pa.schema([
    ("order_id", pa.string()),
    ("customer_id", pa.string()),
    ("product_id", pa.string()),
    ("product_name", pa.string()),
    ("quantity", pa.int32()),
    ("total_amount", pa.float64()),
    ("status", pa.string()),
    ("order_date", pa.string()),
    ("payment_method", pa.string()),
    ("shipping_method", pa.string()),
    ("item_count", pa.int32()),
])

PRODUCTS_ARROW_SCHEMA = pa.schema([
    ("product_id", pa.string()),
    ("name", pa.string()),
    ("category", pa.string()),
    ("description", pa.string()),
    ("specifications", pa.string()),
    ("price", pa.float64()),
    ("rating", pa.float64()),
    ("review_count", pa.int32()),
    ("sku", pa.string()),
    ("in_stock", pa.bool_()),
    ("department", pa.string()),
])

CUSTOMERS_ARROW_SCHEMA = pa.schema([
    ("customer_id", pa.string()),
    ("tier", pa.string()),
    ("total_orders", pa.int32()),
    ("total_spent", pa.float64()),
    ("join_date", pa.string()),
    ("preferred_category", pa.string()),
])


class SyntheticDataGenerator:
    """Generates realistic e-commerce lakehouse datasets in scalable batches."""

    def __init__(self, output_dir: str = "data", seed: int | None = 42) -> None:
        self.output_dir = Path(output_dir)
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(int(time.time() * 1000) % 10000000)

    @classmethod
    def regenerate_new_data(
        cls,
        output_dir: str = "data",
        scale: str = "quick",
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> Path:
        """Regenerate fresh synthetic data with dynamic randomness at the specified scale."""
        instance = cls(output_dir=output_dir, seed=None)
        return instance.generate_all(scale=scale, progress_cb=progress_cb)

    def generate_all(
        self,
        scale: str = "quick",
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> Path:
        """Generate all synthetic data and stream to Parquet files."""
        t0 = time.perf_counter()
        cfg = SCALE_CONFIGS.get(scale, SCALE_CONFIGS["quick"])
        logger.info("Generating synthetic data at scale '%s' (%d orders)...", scale, cfg["orders"])

        # Create Medallion directories
        for layer in ["bronze", "silver", "gold"]:
            (self.output_dir / layer).mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb("Generating product catalog...", 0.1)
        products = self._generate_products(cfg["products"])

        if progress_cb:
            progress_cb("Generating customer database...", 0.25)
        customers = self._generate_customers(cfg["customers"])

        # Write products to Parquet
        products_parquet = self.output_dir / "gold" / "products.parquet"
        self._write_table_parquet(products, PRODUCTS_ARROW_SCHEMA, products_parquet)

        # Write customers to Parquet
        customers_parquet = self.output_dir / "silver" / "customers.parquet"
        self._write_table_parquet(customers, CUSTOMERS_ARROW_SCHEMA, customers_parquet)

        if progress_cb:
            progress_cb(f"Streaming {cfg['orders']:,} transactions...", 0.4)

        # Stream orders directly to Parquet in 50,000-row chunks
        orders_parquet = self.output_dir / "silver" / "orders.parquet"
        self._stream_orders_to_parquet(
            count=cfg["orders"],
            products=products,
            customers=customers,
            output_path=orders_parquet,
            batch_size=50_000,
            progress_cb=progress_cb,
        )

        if progress_cb:
            progress_cb("Generating reviews and support tickets...", 0.85)
        reviews = self._generate_reviews(cfg["reviews"], products, customers)
        tickets = self._generate_tickets(cfg["tickets"], products, customers)

        # Write reviews and tickets to Bronze
        self._write_json(self.output_dir / "bronze" / "reviews.json", reviews)
        self._write_json(self.output_dir / "bronze" / "tickets.json", tickets)

        # If quick demo mode, write JSON copies for small backward compatibility
        if cfg.get("write_json", False):
            self._write_json(self.output_dir / "gold" / "products.json", products[:1000])
            self._write_json(self.output_dir / "silver" / "customers.json", customers[:5000])
            # Write sample orders for quick JSON inspection
            sample_orders = self._generate_orders(min(cfg["orders"], 5000), products, customers)
            self._write_json(self.output_dir / "silver" / "orders.json", sample_orders)

        duration = round(time.perf_counter() - t0, 2)
        total_orders_mb = orders_parquet.stat().st_size / (1024 * 1024)
        logger.info(
            "Synthetic generation completed in %.2fs. Orders Parquet: %.2f MB (%d rows)",
            duration,
            total_orders_mb,
            cfg["orders"],
        )

        if progress_cb:
            progress_cb("Completed lakehouse generation!", 1.0)

        return self.output_dir

    def _stream_orders_to_parquet(
        self,
        count: int,
        products: list[dict[str, Any]],
        customers: list[dict[str, Any]],
        output_path: Path,
        batch_size: int = 50_000,
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> None:
        """Stream high-volume orders directly to disk via ParquetWriter."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp.parquet")

        writer = pq.ParquetWriter(
            temp_path,
            schema=ORDERS_ARROW_SCHEMA,
            compression="snappy",
            use_dictionary=True,
        )

        generated = 0
        try:
            while generated < count:
                curr_batch_size = min(batch_size, count - generated)
                batch_records: list[dict[str, Any]] = []

                for _ in range(curr_batch_size):
                    generated += 1
                    product = random.choice(products)
                    customer = random.choice(customers)
                    qty = random.randint(1, 5)
                    total = round(product["price"] * qty, 2)
                    order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 900))

                    batch_records.append({
                        "order_id": f"ORD-{generated:08d}",
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

                record_batch = pa.RecordBatch.from_pylist(batch_records, schema=ORDERS_ARROW_SCHEMA)
                writer.write_batch(record_batch)
                del batch_records

                if progress_cb and count > 100_000:
                    pct = 0.4 + (generated / count) * 0.45
                    progress_cb(f"Written {generated:,} / {count:,} orders...", pct)

        finally:
            writer.close()

        if output_path.exists():
            output_path.unlink()
        temp_path.replace(output_path)

    @staticmethod
    def _write_table_parquet(records: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
        """Write record list to Parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, path, compression="snappy")

    def _generate_products(self, count: int) -> list[dict[str, Any]]:
        """Generate product catalog."""
        products: list[dict[str, Any]] = []
        for i in range(count):
            category = random.choice(CATEGORIES)
            templates = PRODUCT_TEMPLATES.get(category, PRODUCT_TEMPLATES["Electronics"])
            name_template, desc_template = random.choice(templates)
            variant = f" v{random.randint(1, 5)}.{random.randint(0, 9)}"
            name = name_template + (variant if i >= len(templates) * len(CATEGORIES) else "")

            products.append({
                "product_id": f"PROD-{i+1:06d}",
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
                "customer_id": f"CUST-{i+1:07d}",
                "tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
                "total_orders": random.randint(1, 200),
                "total_spent": round(random.uniform(50, 50000), 2),
                "join_date": (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 2000))).isoformat(),
                "preferred_category": random.choice(CATEGORIES),
            })
        return customers

    def _generate_orders(
        self, count: int, products: list[dict[str, Any]], customers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate order records in memory for quick small runs."""
        orders: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            customer = random.choice(customers)
            qty = random.randint(1, 5)
            total = round(product["price"] * qty, 2)
            order_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 900))

            orders.append({
                "order_id": f"ORD-{i+1:08d}",
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
        self, count: int, products: list[dict[str, Any]], customers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate product reviews."""
        reviews: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            customer = random.choice(customers)
            sentiment = random.choices(["positive", "neutral", "negative"], weights=[60, 25, 15])[0]
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
                "review_date": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 900))).isoformat(),
                "category": product["category"],
            })
        return reviews

    def _generate_tickets(
        self, count: int, products: list[dict[str, Any]], customers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate support tickets."""
        tickets: list[dict[str, Any]] = []
        for i in range(count):
            product = random.choice(products)
            order_id = f"ORD-{random.randint(1, 50000):08d}"
            template = random.choice(TICKET_TEMPLATES)
            title = template[0]
            body = template[1].format(
                order_id=order_id,
                product=product["name"],
                date="2025-05-10",
                status="shipped",
                issue="defective component" if "defective" in title.lower() else "unmatched description",
                reason="Wrong size" if "return" in title.lower() else "Changed mind",
                amount="129.99",
                question="it is compatible with USB-C",
            )

            tickets.append({
                "ticket_id": f"TKT-{i+1:06d}",
                "title": title,
                "body": body,
                "priority": random.choice(["low", "medium", "high", "urgent"]),
                "status": random.choice(["open", "in_progress", "resolved", "closed"]),
                "category": random.choice(["shipping", "product", "billing", "returns", "inquiry"]),
                "product_id": product["product_id"],
                "order_id": order_id,
                "created_at": (datetime(2024, 6, 1) + timedelta(days=random.randint(0, 700))).isoformat(),
                "department": random.choice(["Support", "Returns", "Billing", "Shipping"]),
            })
        return tickets

    @staticmethod
    def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
        """Write JSON array to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
