import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import bittensor as bt
import numpy as np

from gas.cache.content_db import ContentDB, SOURCE_TYPE_TO_DB_NAME_FIELD
from gas.cache.media_storage import MediaStorage
from gas.cache.types import Media, MediaEntry, PromptEntry
from gas.cache.util import extract_media_info, get_format_from_content
from gas.types import MediaType, Modality, SourceType, SOURCE_TYPE_TO_NAME
from gas.utils.huggingface_uploads import upload_images_to_hf, upload_videos_to_hf


class ContentManager:
	"""
	Unified interface for managing content storage and retrieval.
    Encapsulates both filesystem and database operations,
	Provides high-level methods like write_media() and write_prompt().
	"""

	def __init__(
		self,
		base_dir: Optional[Path] = None,
		max_per_source: int = 500,
		enable_source_limits: bool = True,
		prune_strategy: str = "oldest",
		remove_on_sample: bool = True,
		min_source_threshold: float = 0.8,
		min_prompts_threshold: int = 100,
	):
		"""
		Initialize the content manager.

		Args:
			base_dir: Base directory for cache storage (defaults to ~/.cache/sn34)
			max_per_source: Maximum items per source (dataset/scraper/model)
			enable_source_limits: Whether to enable source limits
			prune_strategy: Strategy for pruning ('oldest', 'least_used', 'random')
			remove_on_sample: Whether to remove items when sampled
			min_source_threshold: Minimum items before triggering download
			min_prompts_threshold: Minimum prompts to keep per modality when sampling with remove=True
		"""
		if base_dir is None:
			base_dir = Path("~/.cache/sn34").expanduser()

		self.base_dir = Path(base_dir)
		self.content_db = ContentDB(self.base_dir / "prompts.db")
		self.media_storage = MediaStorage(self.base_dir)

		self.max_per_source = max_per_source
		self.enable_source_limits = enable_source_limits
		self.prune_strategy = prune_strategy
		
		self.remove_on_sample = remove_on_sample
		min_source_threshold = 0.8 if min_source_threshold is None else min_source_threshold
		self.min_source_threshold = int(max_per_source * float(min_source_threshold))
		self.min_prompts_threshold = min_prompts_threshold

	def write_prompt(self, content: str, content_type: str = "prompt", source_media_id: Optional[str] = None, modality: Optional[str] = None) -> str:
		try:
			prompt_id = self.content_db.add_prompt_entry(
				content=content,
				content_type=content_type,
				source_media_id=source_media_id,
				modality=modality
			)
			bt.logging.debug(f"Added {content_type} (modality={modality}) to database with ID: {prompt_id}")
			return prompt_id
		except Exception as e:
			bt.logging.error(f"Error writing {content_type} to database: {e}")
			raise

	def write_generated_media(
		self,
		modality: Modality,
		media_type: MediaType,
		model_name: str,
		prompt_id: str,
		media_content: Any,
		mask_content: Optional[np.ndarray] = None,
		generation_args: Optional[Dict[str, Any]] = None,
	) -> Optional[str]:
		"""
		Write generated media (from models) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL, MediaType.SYNTHETIC, or MediaType.SEMISYNTHETIC
			model_name: Name of the model that generated this media
			prompt_id: ID of the associated prompt
			media_content: The media content (PIL Image, video frames, etc.)
			mask_content: Optional mask for images
			generation_args: args used in generating this media
		Returns:
			Path to the saved media file, or None if failed
		"""
		try:
			format = get_format_from_content(media_content, modality)

			media_data = Media(
				modality=modality,
				media_type=media_type,
				prompt_id=prompt_id,
				media_content=media_content,
				format=format,
				model_name=model_name,
				mask_content=mask_content,
				generation_args=generation_args
			)

			save_path, mask_path = self.media_storage.write_media(media_data)

			if save_path is None:
				bt.logging.error("Failed to write media to filesystem")
				return None

			resolution, file_size = extract_media_info(save_path, media_data.modality)

			media_id = self.content_db.add_media_entry(
				prompt_id=media_data.prompt_id,
				file_path=save_path,
				modality=media_data.modality,
				media_type=media_data.media_type,
				source_type=SourceType.GENERATED,
				generation_args=generation_args,
				model_name=media_data.model_name,
				mask_path=mask_path,
				timestamp=int(time.time()),
				resolution=resolution,
				file_size=file_size,
				format=media_data.format,
			)

			bt.logging.info(f"Saved media to {save_path} with database ID: {media_id}")
			return save_path

		except Exception as e:
			bt.logging.error(f"Error writing media: {e}")
			return None

	def write_miner_media(
		self,
		modality: Modality,
		media_type: MediaType,
		prompt_id: str,
		uid: int,
		hotkey: str,
		media_content: bytes,
		content_type: str,
		task_id: str,
		model_name: Optional[str] = None,
		perceptual_hash: Optional[str] = None,
		c2pa_verified: Optional[bool] = None,
		c2pa_issuer: Optional[str] = None,
	) -> Optional[str]:
		"""
		Write miner-generated binary media to storage.
		Follows the same pattern as other write methods.

		Args:
			uid: Miner UID
			hotkey: Miner hotkey
			binary_data: Raw binary media data
			content_type: MIME content type (e.g., "image/png", "video/mp4")
			task_id: Unique task identifier for filename
			model_name: Optional model name
			perceptual_hash: Pre-computed perceptual hash for duplicate detection
			c2pa_verified: Whether C2PA verification passed
			c2pa_issuer: C2PA issuer name if verified

		Returns:
			Path to saved file if successful, None if failed
		"""
		try:
			media_data = Media(
				modality=modality,
				media_type=media_type,
				prompt_id=prompt_id,
				media_content=media_content,
				format=get_format_from_content(media_content, modality),
				model_name=model_name,
				metadata={"uid": uid, "task_id": task_id, "source": "miner"}
			)

			save_path, mask_path = self.media_storage.write_media(media_data)
			if save_path is None:
				bt.logging.error("Failed to write miner media to filesystem")
				return None

			resolution, file_size = extract_media_info(save_path, modality)
			media_id = self.content_db.add_media_entry(
				prompt_id=prompt_id,
				file_path=save_path,
				modality=modality,
				media_type=media_type,
				source_type=SourceType.MINER,
				uid=uid,
				hotkey=hotkey,
				model_name=model_name,
				verified=False,  # Initially unverified
				timestamp=int(time.time()),
				resolution=resolution,
				file_size=file_size,
				format=media_data.format,
				perceptual_hash=perceptual_hash,
				c2pa_verified=c2pa_verified,
				c2pa_issuer=c2pa_issuer,
			)

			bt.logging.info(f"Saved miner media to {save_path} with database ID: {media_id}")
			return str(save_path)

		except Exception as e:
			bt.logging.error(f"Error writing miner media: {e}")
			return None

	def write_scraped_media(
		self,
		modality: Modality,
		media_type: MediaType,
		prompt_id: str,
		media_content: Any,
		download_url: str,
		scraper_name: str,
		mask_content: Optional[np.ndarray] = None,
		resolution: Optional[tuple[int, int]] = None,
	) -> Optional[str]:
		"""
		Write scraped media (from web sources) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL or MediaType.SEMISYNTHETIC
			prompt_id: ID of the associated search query
			media_content: The media content (PIL Image, video frames, etc.)
			download_url: URL where the media was scraped from
			scraper_name: Name of the scraper (e.g., 'google', 'bing')
			mask_content: Optional mask for images
			resolution: Optional (width, height) tuple

		Returns:
			Path to the saved media file, or None if failed
		"""
		media_data = Media(
			modality=modality,
			media_type=media_type,
			prompt_id=prompt_id,
			media_content=media_content,
			format=get_format_from_content(media_content, modality),
			model_name=None,  # Not applicable for scraped media
			mask_content=mask_content,
		)

		save_path, mask_path = self.media_storage.write_media(media_data)

		if save_path is None:
			bt.logging.error("Failed to write scraped media to filesystem")
			return None

		# Use provided resolution if available, otherwise extract from file
		if resolution is None:
			resolution, file_size = extract_media_info(save_path, media_data.modality)
		else:
			file_size = extract_media_info(save_path, media_data.modality)[1]

		media_id = self.content_db.add_media_entry(
			prompt_id=media_data.prompt_id,
			file_path=save_path,
			modality=media_data.modality,
			media_type=media_data.media_type,
			source_type=SourceType.SCRAPER,
			download_url=download_url,
			scraper_name=scraper_name,
			mask_path=mask_path,
			timestamp=int(time.time()),
			resolution=resolution,
			file_size=file_size,
			format=media_data.format,
		)

		bt.logging.info(
			f"Saved scraped media to {save_path} with database ID: {media_id}"
		)
		return save_path

	def write_dataset_media(
		self,
		modality: Modality,
		media_type: MediaType,
		media_content: Any,
		dataset_name: str,
		dataset_source_file: str,
		dataset_index: str,
		mask_content: Optional[np.ndarray] = None,
		resolution: Optional[tuple[int, int]] = None,
	) -> Optional[str]:
		"""
		Write dataset media (from HuggingFace datasets) to storage.

		Args:
			modality: Modality.IMAGE or Modality.VIDEO
			media_type: MediaType.REAL (datasets are real media)
			media_content: The media content (bytes, PIL Image, etc.)
			dataset_name: Name/path of the dataset (e.g., 'laion/laion2B-en')
			dataset_source_file: Source file within dataset (e.g., 'data_001.parquet')
			dataset_index: Index within the dataset file
			mask_content: Optional mask for images
			resolution: Optional (width, height) tuple

		Returns:
			Path to the saved media file, or None if failed
		"""
		media_data = Media(
			modality=modality,
			media_type=media_type,
			prompt_id=None,  # Dataset media is not tied to prompts
			media_content=media_content,
			format=get_format_from_content(media_content, modality),
			model_name=None,  # Not applicable for dataset media
			mask_content=mask_content,
		)

		save_path, mask_path = self.media_storage.write_media(media_data)

		if save_path is None:
			bt.logging.error("Failed to write dataset media to filesystem")
			return None

		# Use provided resolution if available, otherwise extract from file
		if resolution is None:
			resolution, file_size = extract_media_info(save_path, media_data.modality)
		else:
			file_size = extract_media_info(save_path, media_data.modality)[1]

		# Add entry to database with source_type='dataset'
		media_id = self.content_db.add_media_entry(
			prompt_id=None,  # Dataset media is not tied to prompts
			file_path=save_path,
			modality=media_data.modality,
			media_type=media_data.media_type,
			source_type=SourceType.DATASET,
			dataset_name=dataset_name,
			dataset_source_file=dataset_source_file,
			dataset_index=dataset_index,
			mask_path=mask_path,
			timestamp=int(time.time()),
			resolution=resolution,
			file_size=file_size,
			format=media_data.format,
		)

		return save_path

	def sample_prompts(
        self, k: 
        int = 1, 
        remove: bool = False, 
        strategy: str = "random",
        modality: Optional[str] = None
	) -> List[PromptEntry]:
		"""
		Sample prompts from the database.
		
		Args:
			k: Number of prompts to sample
			remove: If True, deletes sampled prompts (only if min_prompts_threshold remain)
			strategy: Sampling strategy ('random', 'least_used', 'oldest', 'newest')
			modality: Optional modality filter ('image', 'video', 'audio')
		"""
		return self.content_db.sample_prompt_entries(
			k=k, 
			remove=remove, 
			strategy=strategy, 
			content_type="prompt", 
			modality=modality,
			min_prompts_threshold=self.min_prompts_threshold,
		)

	def sample_search_queries(
        self, 
        k: int = 1, 
        remove: bool = False, 
        strategy: str = "random"
	) -> List[PromptEntry]:
		"""
		Sample search queries from the database.
		
		Args:
			k: Number of queries to sample
			remove: If True, deletes sampled queries (only if min_prompts_threshold remain)
			strategy: Sampling strategy ('random', 'least_used', 'oldest', 'newest')
		"""
		return self.content_db.sample_prompt_entries(
			k=k, 
			remove=remove, 
			strategy=strategy, 
			content_type="search_query",
			min_prompts_threshold=self.min_prompts_threshold,
		)

	def get_prompt_by_id(self, prompt_id: str) -> Optional[str]:
		return self.content_db.get_prompt_by_id(prompt_id)

	def sample_media(
		self,
		k: int = 1,
		modality: Modality = Modality.IMAGE,
		media_type: MediaType = MediaType.SYNTHETIC,
		remove: bool = False,
		strategy: str = "random"
	) -> List[MediaEntry]:
		return self.content_db.sample_media_entries(
			k=k,
			modality=modality,
			media_type=media_type,
			strategy=strategy,
			remove=remove
		)

	def sample_media_with_content(
		self,
		modality: Modality,
		media_type: MediaType,
		count: int = 1,
		remove_from_cache: bool = None,
		**kwargs,
	) -> Optional[Dict[str, Any]]:
        
		should_remove = remove_from_cache if remove_from_cache is not None else self.remove_on_sample
		media_entries = self.content_db.sample_media_entries(
			k=count,
			modality=modality,
			media_type=media_type,
			strategy=kwargs.get("strategy", "random"),
			remove=False,
		)
		if not media_entries:
			print(f"No media available in database for {modality}/{media_type}")
			return {'count': 0, 'items': []}

		# Split by source_type so we only remove dataset/scraper on sample
		generated_entries: List[MediaEntry] = [
			e for e in media_entries 
			if getattr(e, 'source_type', None) == SourceType.GENERATED
		]
		non_generated_entries: List[MediaEntry] = [
			e for e in media_entries
			if getattr(e, 'source_type', None) != SourceType.GENERATED
		]

		items: List[Dict[str, Any]] = []
		if non_generated_entries:
			non_gen_items = self.media_storage.retrieve_media(
				media_entries=non_generated_entries,
				modality=modality,
				remove_from_cache=False,
				**kwargs,
			)['items']
			items.extend(non_gen_items)

			if should_remove:
				for entry in non_generated_entries:
					self.delete_media(file_path=entry.file_path)

		if generated_entries:
			gen_items = self.media_storage.retrieve_media(
				media_entries=generated_entries,
				modality=modality,
				remove_from_cache=False,
				**kwargs,
			)['items']
			items.extend(gen_items)

		combined_entries: List[MediaEntry] = non_generated_entries + generated_entries
		for media, db_entry in zip(items, combined_entries):
			media['id'] = db_entry.id
			media['metadata'] = db_entry.to_dict()

			origin_field = SOURCE_TYPE_TO_NAME[db_entry.source_type]
			origin_value = getattr(db_entry, origin_field)
			media['source_type'] = db_entry.source_type
			media['source_name'] = origin_value

		return {'count': len(items), 'items': items}

	def sample_prompts_with_source_media(
		self,
		k: int = 1,
		remove: bool = True,
		strategy: str = "random",
		**kwargs
	) -> List[Dict[str, Any]]:
		"""
		Sample prompts and load their associated source media content (first item only).
		Returns a list of { 'prompt': PromptEntry, 'media': media_item }.
		"""
		prompt_entries = self.content_db.sample_prompt_entries(
			k=k, 
			remove=remove, 
			strategy=strategy, 
			content_type="prompt",
			min_prompts_threshold=self.min_prompts_threshold,
		)
		results: List[Dict[str, Any]] = []
		for prompt in prompt_entries:
			if not prompt.source_media_id:
				continue
			media_entries = self.content_db.get_media_entries(media_id=prompt.source_media_id)
			if not media_entries:
				continue
			media_content = self.media_storage.retrieve_media(
				media_entries=media_entries,
				modality=Modality.IMAGE,
				**kwargs
			)
			if media_content.get("count", 0) > 0:
				results.append({
					"prompt": prompt,
					"media": media_content["items"][0]
				})
		return results

	def _check_and_prune_source(self, source_type: SourceType, source_name: str, max_count: int) -> int:
		if not self.enable_source_limits:
			return 0
		current_count = self.content_db.get_source_count(source_type, source_name)
		if current_count >= max_count:
			pruned = self.content_db.prune_source_media(
				source_type, source_name, max_count, self.prune_strategy
			)
			if pruned > 0:
				bt.logging.info(f"Pruned {pruned} items from {source_type.value} '{source_name}' to enforce cap {max_count}")
			return pruned
		return 0

	def enforce_source_caps(self) -> Dict[str, int]:
		results: Dict[str, int] = {}
		if not self.enable_source_limits:
			return results
		try:
			counts = self.content_db.get_source_counts()
			for source_type_str, sources in counts.items():
				st = SourceType(source_type_str)
				for source_name, count in sources.items():
					if count > self.max_per_source:
						pruned = self._prune_source_media(st, source_name, self.max_per_source)
						if pruned > 0:
							key = f"{st.value}:{source_name}"
							results[key] = pruned
							bt.logging.info(
								f"[CONTENT] Enforced cap: pruned {pruned} from {st.value} '{source_name}' "
								f"(count {count} -> ≤ {self.max_per_source})"
							)
		except Exception as e:
			bt.logging.error(f"Error enforcing source caps: {e}")
		return results

	def _prune_source_media(self, source_type: SourceType, source_name: str, max_count: int) -> int:

		col = SOURCE_TYPE_TO_DB_NAME_FIELD.get(source_type)
		if not col:
			return 0

		current = self.content_db.get_source_count(source_type, source_name)
		if current <= max_count:
			return 0

		to_remove = current - max_count
		order_clause = 'created_at ASC' if self.prune_strategy in ('oldest', 'least_used') else 'RANDOM()'

		with self.content_db._get_db_connection() as conn:
			cursor = conn.execute(
				f"""
				SELECT file_path FROM media
				WHERE source_type = ? AND {col} = ?
				ORDER BY {order_clause}
				LIMIT ?
				""",
				(source_type.value, source_name, to_remove),
			)
			file_paths = [row[0] for row in cursor.fetchall()]

		deleted_count = 0
		for file_path in file_paths:
			if self.delete_media(file_path=file_path):
				deleted_count += 1

		return deleted_count

	def get_media_entry_by_file_path(self, file_path: str) -> Optional[MediaEntry]:
		return self.content_db.get_media_entry_by_file_path(file_path)

	def delete_media(self, file_path: str = None, media_id: str = None) -> bool:
		if not file_path and not media_id:
			return False

		try:
			if media_id and not file_path:
				media_entry = self.content_db.get_media_entries(media_id=media_id)
				if not media_entry:
					return False
				file_path = media_entry[0].file_path

			file_path_obj = Path(file_path)
			if file_path_obj.exists():
				success = self.media_storage.delete_media_file(file_path_obj)
				if not success:
					return False

			return self.content_db.delete_media_entry_by_file_path(file_path)

		except Exception as e:
			bt.logging.error(f"Error deleting media: {e}")
			return False

	def delete_media_by_file_path(self, file_path: str) -> bool:
		return self.delete_media(file_path=file_path)

	def get_miner_media(self, verification_status: Optional[str] = None) -> List[MediaEntry]:
		"""
		Get miner media by verification status.
		
		Args:
			verification_status: Optional verification status filter:
				- 'pending': Not verified yet (verified=0, failed_verification=0)
				- 'verified': Passed verification (verified=1, failed_verification=0)  
				- 'failed': Failed verification (verified=0, failed_verification=1)
				- None: Return all miner media regardless of verification status
		
		Returns:
			List of MediaEntry objects matching the verification status
		"""
		return self.content_db.get_miner_media(verification_status=verification_status)

	def get_pending_verification_count(self) -> int:
		"""Get count of media entries pending verification."""
		pending_media = self.get_miner_media(verification_status="pending")
		return len(pending_media)

	def mark_miner_media_verified(self, media_id: str) -> bool:
		return self.content_db.mark_miner_media_verified(media_id)

	def mark_miner_media_failed_verification(self, media_id: str) -> bool:
		"""Mark miner media as failed verification."""
		return self.content_db.mark_miner_media_failed_verification(media_id)

	def get_unuploaded_media(
		self, 
		limit: int = 100, 
		modality: str = None, 
		source_type: str = None
	) -> List[MediaEntry]:
		return self.content_db.get_unuploaded_media(
			limit=limit,
			modality=modality,
			source_type=source_type
		)

	def mark_media_uploaded(self, media_ids: List[str]) -> bool:
		return self.content_db.mark_media_uploaded(media_ids)

	def mark_media_rewarded(self, media_ids: List[str]) -> bool:
		"""Mark media entries as rewarded."""
		return self.content_db.mark_media_rewarded(media_ids)

	def get_unrewarded_verified_miner_media(self, limit: int = 100) -> List[MediaEntry]:
		"""Get verified miner media entries that haven't been rewarded yet."""
		return self.content_db.get_unrewarded_verified_miner_media(limit=limit)

	def get_recent_verified_miner_media(self, lookback_hours: float = 2.0, limit: int = 1000) -> List[MediaEntry]:
		"""Get verified miner media entries from the last N hours."""
		return self.content_db.get_recent_verified_miner_media(lookback_hours=lookback_hours, limit=limit)

	def get_recent_failed_miner_media(self, lookback_hours: float = 2.0, limit: int = 1000) -> List[MediaEntry]:
		"""Get miner media that failed verification in the last N hours."""
		return self.content_db.get_recent_failed_miner_media(lookback_hours=lookback_hours, limit=limit)

	def get_verification_stats_last_n_hours(
		self, lookback_hours: float = 2.0, limit: int = None
	) -> Dict[str, Dict[str, Any]]:
		"""
		Get verification statistics for miner media in the last N hours: pass and fail
		counts and pass rate. Includes both rewarded and unrewarded verified media.
		Use for generator base rewards (eligibility = verified submission in last N hours).

		Args:
			lookback_hours: Number of hours to look back (default 2.0).
			limit: Maximum number of entries per type to consider (default 1000).

		Returns:
			Dict mapping miner hotkey to verification stats: total_verified (passed),
			total_failed (failed), total_evaluated, pass_rate, media_ids (verified only).
		"""
		try:
			limit_val = limit or 1000
			verified_media = self.get_recent_verified_miner_media(
				lookback_hours=lookback_hours, limit=limit_val
			)
			failed_media = self.get_recent_failed_miner_media(
				lookback_hours=lookback_hours, limit=limit_val
			)
			if not verified_media and not failed_media:
				bt.logging.debug(f"No verified or failed miner media in last {lookback_hours}h")
				return {}
			return self._build_verification_stats_from_verified_and_failed(
				verified_media, failed_media
			)
		except Exception as e:
			bt.logging.error(f"Error getting verification stats for last {lookback_hours}h: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())
			return {}

	def _build_verification_stats_from_verified_and_failed(
		self,
		verified_media: List[MediaEntry],
		failed_media: List[MediaEntry],
	) -> Dict[str, Dict[str, Any]]:
		"""Build verification stats from verified and failed media (e.g. both from last N hours)."""
		miner_stats = {}
		for media in verified_media:
			if not media.hotkey or not media.uid:
				continue
			hotkey = media.hotkey
			if hotkey not in miner_stats:
				miner_stats[hotkey] = {
					"uid": media.uid,
					"verified_media_ids": [],
					"total_verified": 0,
					"total_failed": 0,
					"last_timestamp": media.created_at
				}
			miner_stats[hotkey]["verified_media_ids"].append(media.id)
			miner_stats[hotkey]["total_verified"] += 1
			if media.created_at > miner_stats[hotkey]["last_timestamp"]:
				miner_stats[hotkey]["last_timestamp"] = media.created_at
		for media in failed_media:
			if not media.hotkey or not media.uid:
				continue
			hotkey = media.hotkey
			if hotkey not in miner_stats:
				miner_stats[hotkey] = {
					"uid": media.uid,
					"verified_media_ids": [],
					"total_verified": 0,
					"total_failed": 0,
					"last_timestamp": media.created_at
				}
			miner_stats[hotkey]["total_failed"] += 1
			if media.created_at > miner_stats[hotkey]["last_timestamp"]:
				miner_stats[hotkey]["last_timestamp"] = media.created_at

		verification_stats = {}
		for hotkey, stats in miner_stats.items():
			verified = stats["total_verified"]
			failed = stats["total_failed"]
			total_evaluated = verified + failed
			pass_rate = (verified / total_evaluated) if total_evaluated > 0 else 0.0
			verification_stats[hotkey] = {
				"uid": stats["uid"],
				"total_verified": verified,
				"total_submissions": total_evaluated,
				"total_failed": failed,
				"total_evaluated": total_evaluated,
				"pass_rate": pass_rate,
				"media_ids": stats["verified_media_ids"],
				"last_timestamp": stats["last_timestamp"]
			}
		bt.logging.info(f"Retrieved verification stats for {len(verification_stats)} miners")
		return verification_stats

	def _build_verification_stats_from_verified_media(
		self, verified_media: List[MediaEntry]
	) -> Dict[str, Dict[str, Any]]:
		"""Build verification stats dict from a list of verified media entries (failed from all-time)."""
		miner_stats = {}
		for media in verified_media:
			if not media.hotkey or not media.uid:
				continue
			hotkey = media.hotkey
			if hotkey not in miner_stats:
				miner_stats[hotkey] = {
					"uid": media.uid,
					"verified_media_ids": [],
					"total_verified": 0,
					"total_submissions": 0,
					"total_failed": 0,
					"last_timestamp": media.created_at
				}
			miner_stats[hotkey]["verified_media_ids"].append(media.id)
			miner_stats[hotkey]["total_verified"] += 1
			if media.created_at > miner_stats[hotkey]["last_timestamp"]:
				miner_stats[hotkey]["last_timestamp"] = media.created_at

		for hotkey, stats in miner_stats.items():
			uid = stats["uid"]
			all_miner_media = self.get_miner_media(verification_status=None)
			miner_media = [m for m in all_miner_media if m.hotkey == hotkey and m.uid == uid]
			stats["total_submissions"] = len(miner_media)
			stats["total_failed"] = len([m for m in miner_media if m.failed_verification])

		verification_stats = {}
		for hotkey, stats in miner_stats.items():
			verified = stats["total_verified"]
			failed = stats["total_failed"]
			total_evaluated = verified + failed
			pass_rate = (verified / total_evaluated) if total_evaluated > 0 else 0.0
			verification_stats[hotkey] = {
				"uid": stats["uid"],
				"total_verified": verified,
				"total_submissions": stats["total_submissions"],
				"total_failed": failed,
				"total_evaluated": total_evaluated,
				"pass_rate": pass_rate,
				"media_ids": stats["verified_media_ids"],
				"last_timestamp": stats["last_timestamp"]
			}
		bt.logging.info(f"Retrieved verification stats for {len(verification_stats)} miners")
		return verification_stats

	def get_unrewarded_verification_stats(self, limit: int = None, include_all: bool = False, lookback_hours: float = None) -> Dict[str, Dict[str, Any]]:
		"""
		Get verification statistics for miner media (pass rates, etc.).
		Returns raw statistics without computing rewards - that's done in rewards.py.

		Args:
			limit: Maximum number of entries to consider per miner
			include_all: If False (default), only return stats for unrewarded media.
						If True, return stats for ALL verified media (rewarded + unrewarded)
			lookback_hours: If provided, use time-based lookback instead of rewarded flag.
						   Returns all verified media from the last N hours regardless of
						   rewarded status. This ensures miners with recent verified submissions
						   are always eligible for rewards.

		Returns:
			Dict mapping miner hotkey to verification stats:
			{
				"hotkey": {
					"uid": int,
					"total_verified": int,        # Count of verified media
					"total_submissions": int,     # Count of all submissions (verified + failed + pending)
					"total_failed": int,         # Count of failed verification media
					"total_evaluated": int,      # Count of evaluated media (verified + failed)
					"pass_rate": float,          # verified / (verified + failed)
					"media_ids": List[str],      # IDs of verified media to mark as rewarded
					"last_timestamp": float      # Creation timestamp of most recent verified media sample
				}
			}
		"""
		try:
			if lookback_hours is not None:
				verified_media = self.get_recent_verified_miner_media(
					lookback_hours=lookback_hours, limit=limit or 1000
				)
				unrewarded_media = self.get_unrewarded_verified_miner_media(limit=limit or 1000)
				if len(unrewarded_media) == 0 and len(verified_media) > 0:
					bt.logging.warning(
						f"Time-based lookback found {len(verified_media)} verified media, "
						f"but rewarded-flag approach found 0. This indicates a timing edge case."
					)
				elif len(unrewarded_media) != len(verified_media):
					bt.logging.debug(
						f"Time-based: {len(verified_media)} media, rewarded-flag: {len(unrewarded_media)} media"
					)
			elif include_all:
				verified_media = self.get_miner_media(verification_status="verified")
				if limit and len(verified_media) > limit:
					verified_media = verified_media[:limit]
			else:
				verified_media = self.get_unrewarded_verified_miner_media(limit=limit or 1000)
			
			if not verified_media:
				bt.logging.debug("No verified miner media found")
				return {}
			return self._build_verification_stats_from_verified_media(verified_media)

		except Exception as e:
			bt.logging.error(f"Error getting unrewarded verification stats: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())
			return {}

	def upload_batch_to_huggingface(
		self, 
		hf_token: str, 
		hf_dataset_repos: dict, 
		upload_batch_size: int, 
		images_per_archive: int,
		videos_per_archive: int,
		validator_hotkey: str = None,
		validator_uid: int = None,
		num_batches: int = 1
	):
		"""Upload unuploaded media from database to HuggingFace, separated by source (miner vs validator) and modality"""
		try:
			if num_batches is None or num_batches < 1:
				bt.logging.warning(f"Invalid num_batches value: {num_batches}, using default of 1")
				num_batches = 1
			
			total_uploaded_all_batches = 0

			for batch_num in range(num_batches):
				bt.logging.info(f"Processing upload batch {batch_num + 1}/{num_batches}")

				# Prioritize verified miner media, fill remaining with validator-generated media
				media_by_modality = {}
				total_found = 0

				for modality in ["image", "video"]:
					# First, get verified miner media up to batch size
					verified_miner_media = self.content_db.get_unuploaded_media(
						limit=upload_batch_size, 
						modality=modality, 
						source_type='miner'
					)

					# Fill remaining slots with validator-generated media
					remaining_slots = max(0, upload_batch_size - len(verified_miner_media))
					validator_media = []
					if remaining_slots > 0:
						validator_media = self.content_db.get_unuploaded_media(
							limit=remaining_slots, 
							modality=modality, 
							source_type='generated'
						)

					# Combine: miner first, then validator
					combined_media = verified_miner_media + validator_media
					media_by_modality[modality] = combined_media
					total_found += len(combined_media)

					bt.logging.info(
						f"{modality}: {len(verified_miner_media)} miner + "
						f"{len(validator_media)} validator = {len(combined_media)} total"
					)

				if total_found == 0:
					bt.logging.info(f"No more unuploaded media found after {batch_num} batches")
					break

				bt.logging.info(
					f"Batch {batch_num + 1}: Found {total_found} unuploaded media files to upload "
					f"({len(media_by_modality['image'])} images, {len(media_by_modality['video'])} videos)"
				)

				all_successfully_processed_ids = []

				# Upload images (mixed miner + validator)
				if media_by_modality['image']:
					uploaded_ids = upload_images_to_hf(
						media_entries=media_by_modality['image'],
						hf_token=hf_token,
						dataset_repo=hf_dataset_repos['image'],
						images_per_archive=images_per_archive,
						validator_hotkey=validator_hotkey,
						validator_uid=validator_uid
					)
					all_successfully_processed_ids.extend(uploaded_ids)

				# Upload videos (mixed miner + validator)
				if media_by_modality['video']:
					uploaded_ids = upload_videos_to_hf(
						media_entries=media_by_modality['video'],
						hf_token=hf_token,
						dataset_repo=hf_dataset_repos['video'],
						videos_per_archive=videos_per_archive,
						validator_hotkey=validator_hotkey,
						validator_uid=validator_uid
					)
					all_successfully_processed_ids.extend(uploaded_ids)

				# Mark all successfully uploaded media as uploaded in db
				if all_successfully_processed_ids:
					success = self.mark_media_uploaded(all_successfully_processed_ids)
					if success:
						bt.logging.info(f"Batch {batch_num + 1}: Marked {len(all_successfully_processed_ids)} entries as uploaded in database")
						total_uploaded_all_batches += len(all_successfully_processed_ids)
					else:
						bt.logging.warning(f"Batch {batch_num + 1}: Failed to mark media as uploaded in database")

			if total_uploaded_all_batches > 0:
				bt.logging.info(f"✅ Upload cycle complete: {total_uploaded_all_batches} total files uploaded across {batch_num + 1} batches")

		except Exception as e:
			bt.logging.error(f"Error uploading batch to HuggingFace: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())

	def get_dataset_media_counts(self) -> Dict[str, int]:
		"""
		Returns:
			Dictionary with counts for each modality/media_type combination
		"""
		return self.content_db.get_dataset_media_counts()

	def get_sources_needing_data(self) -> Dict[str, List[str]]:
		"""
		Returns:
			Dictionary with source types as keys and lists of source names that need data
		"""
		source_stats = self.content_db.get_source_counts()
		sources_needing_data: Dict[str, List[str]] = {}
		for source_type, sources in source_stats.items():
			sources_needing_data[source_type] = []
			for source_name, count in sources.items():
				if count < self.min_source_threshold:
					sources_needing_data[source_type].append(source_name)
		return sources_needing_data

	def get_source_count(self, source_type: SourceType, source_name: str) -> int:
		"""
		Args:
			source_type: SourceType.DATASET, SourceType.SCRAPER, or SourceType.GENERATED
			source_name: Name of the dataset, scraper, or model

		Returns:
			Count of media items for this source
		"""
		return self.content_db.get_source_count(source_type, source_name)

	def needs_more_data(self, source_type: SourceType, source_name: str) -> bool:
		"""
		Args:
			source_type: SourceType.DATASET, SourceType.SCRAPER, or SourceType.GENERATED
			source_name: Name of the source

		Returns:
			True if the source needs more data
		"""
		return self.get_source_count(source_type, source_name) < self.min_source_threshold

	def check_duplicate(
		self,
		perceptual_hash: str,
		threshold: int = 8,
		limit: int = 1000,
		prompt_id: Optional[str] = None,
	) -> Optional[tuple]:
		"""
		Check if a perceptual hash has duplicates in the database.

		Args:
			perceptual_hash: Hash to check for duplicates
			threshold: Maximum Hamming distance to consider as duplicate
			limit: Maximum number of hashes to check
			prompt_id: If provided, only check duplicates within this prompt

		Returns:
			Tuple of (media_id, hamming_distance) for closest match, or None if no duplicate
		"""
		from gas.verification.duplicate_detection import check_duplicate_in_db
		return check_duplicate_in_db(self.content_db, perceptual_hash, threshold, limit, prompt_id=prompt_id)

	def compute_and_check_duplicate(
		self,
		media_content: bytes,
		modality: str,
		threshold: int = 8,
	) -> tuple:
		"""
		Compute perceptual hash for media and check for duplicates.

		Args:
			media_content: Raw media bytes
			modality: "image" or "video"
			threshold: Hamming distance threshold for duplicates

		Returns:
			Tuple of (perceptual_hash, is_duplicate, duplicate_info)
			where duplicate_info is (media_id, distance) if duplicate found, else None
		"""
		from gas.verification.duplicate_detection import compute_media_hash

		perceptual_hash = compute_media_hash(media_content, modality=modality)
		if not perceptual_hash:
			return None, False, None

		duplicate_info = self.check_duplicate(perceptual_hash, threshold)
		is_duplicate = duplicate_info is not None

		return perceptual_hash, is_duplicate, duplicate_info

	def cleanup_uploaded_media(
		self,
		min_age_hours: float = 24.0,
		require_rewarded: bool = True,
		batch_size: int = 1000,
	) -> Dict[str, int]:
		"""
		Clean up media that has been uploaded to HuggingFace (and optionally rewarded).
		Deletes both database entries and files from disk.

		Args:
			min_age_hours: Minimum age in hours before media can be cleaned up
			require_rewarded: If True, only delete media that is both uploaded AND rewarded
			batch_size: Maximum number of entries to delete per call

		Returns:
			Dict with cleanup statistics: {'media_deleted', 'prompts_deleted', 'files_deleted'}
		"""
		total_media = 0
		total_prompts = 0
		total_files = 0

		try:
			while True:
				media_deleted, prompts_deleted, file_paths = self.content_db.cleanup_uploaded_media(
					min_age_hours=min_age_hours,
					require_rewarded=require_rewarded,
					batch_size=batch_size,
				)

				if media_deleted == 0:
					break

				total_media += media_deleted
				total_prompts += prompts_deleted

				# Delete files from disk
				for file_path in file_paths:
					try:
						path = Path(file_path)
						if path.exists():
							path.unlink()
							total_files += 1
					except Exception as e:
						bt.logging.warning(f"[CLEANUP] Failed to delete file {file_path}: {e}")

				bt.logging.info(
					f"[CLEANUP] Batch complete: {media_deleted} media, {prompts_deleted} prompts, "
					f"{len(file_paths)} files"
				)

			if total_media > 0:
				bt.logging.success(
					f"[CLEANUP] Total cleaned: {total_media} media entries, "
					f"{total_prompts} orphaned prompts, {total_files} files from disk"
				)

		except Exception as e:
			bt.logging.error(f"[CLEANUP] Error during cleanup: {e}")
			import traceback
			bt.logging.error(traceback.format_exc())

		return {
			'media_deleted': total_media,
			'prompts_deleted': total_prompts,
			'files_deleted': total_files,
		}
